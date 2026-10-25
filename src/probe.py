"""Execution-time, fixed-output counterfactual tool identifier scoring.

These scores measure model sensitivity, not whether a call is authorized. The
two margins need not separate attacks from legitimate data dependencies. Only
the proposed identifier is scored; arguments require an independent policy.
``joint_loglik`` is the sum of token log probabilities. Its length-normalized
counterpart is a ranking statistic, not the log probability of the identifier.

The module's masks, aggregation and reference logit scorer use only the standard
library. Torch is imported lazily by HFSequenceScorer; dependencies and model
versions belong to the caller's frozen experiment configuration.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


VIEW_NAMES = ("original", "observation_masked", "user_masked")


class ProbeError(ValueError):
    """An invalid scoring event is retained as an error, never silently fixed."""


class TokenBoundaryError(ProbeError):
    """Joint tokenization changes the fixed prefix or proposed identifier."""


@dataclass(frozen=True)
class ContentSpan:
    """Half-open character range in one message's *content*, before masking."""

    message_index: int
    start: int
    end: int
    expected_text: str | None = None


def latest_tool_observation_span(messages: Sequence[Mapping[str, Any]]) -> ContentSpan:
    """Select only the last native tool response body, preserving its metadata."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "tool":
            body = messages[index].get("content")
            if not isinstance(body, str):
                raise ProbeError("Tool observation content must be a text string")
            return ContentSpan(index, 0, len(body), body)
    raise ProbeError("No native tool response; provide an explicit observation body span")


def _validate_span(messages, span, allowed_roles):
    if not isinstance(span, ContentSpan):
        raise ProbeError("Mask ranges must be explicit ContentSpan instances")
    if not 0 <= span.message_index < len(messages):
        raise ProbeError("Mask message index is out of range")
    message = messages[span.message_index]
    if message.get("role") not in allowed_roles:
        raise ProbeError("Mask cannot alter the system or an unselected message role")
    body = message.get("content")
    if not isinstance(body, str) or not 0 <= span.start <= span.end <= len(body):
        raise ProbeError("Mask content range is invalid")
    if span.expected_text is not None and body[span.start:span.end] != span.expected_text:
        raise ProbeError("Mask span no longer matches the recorded source text")


def _mask_spans(messages, spans, placeholder, allowed_roles):
    if not isinstance(placeholder, str):
        raise ProbeError("Mask placeholder must be text")
    result = copy.deepcopy(list(messages))
    by_message = {}
    provenance = []
    for span in spans:
        _validate_span(messages, span, allowed_roles)
        by_message.setdefault(span.message_index, []).append(span)
        removed = messages[span.message_index]["content"][span.start:span.end]
        provenance.append({"message_index": span.message_index,
                           "role": messages[span.message_index]["role"],
                           "start": span.start, "end": span.end,
                           "removed_sha256": hashlib.sha256(removed.encode()).hexdigest(),
                           "removed_characters": len(removed), "replacement": placeholder})
    for index, selected in by_message.items():
        ordered = sorted(selected, key=lambda span: (span.start, span.end))
        if any(left.end > right.start for left, right in zip(ordered, ordered[1:])):
            raise ProbeError("Overlapping mask ranges are ambiguous")
        body = result[index]["content"]
        for span in reversed(ordered):
            body = body[:span.start] + placeholder + body[span.end:]
        result[index]["content"] = body
    return result, provenance


def make_counterfactual_views(
    messages: Sequence[Mapping[str, Any]],
    *,
    user_spans: Sequence[ContentSpan],
    observation_span: ContentSpan | None = None,
    observation_placeholder: str = "[OBSERVATION_OMITTED]",
    user_placeholder: str = "[USER_INSTRUCTION_OMITTED]",
) -> dict:
    """Build three independently masked views without deleting message structure.

    User ranges are mandatory: a user message can also contain tool definitions
    and a ReAct scratchpad. Native tool responses can be selected automatically;
    a mixed ReAct user message needs an explicit observation body range.
    """
    if not user_spans:
        raise ProbeError("At least one user instruction span must be selected")
    observation_span = observation_span or latest_tool_observation_span(messages)
    for selected in user_spans:
        _validate_span(messages, selected, {"user"})
        if selected.message_index == observation_span.message_index and (
            max(selected.start, observation_span.start) < min(selected.end, observation_span.end)
        ):
            raise ProbeError("Observation and user instruction ranges must not overlap")
    observation, observation_provenance = _mask_spans(
        messages, [observation_span], observation_placeholder, {"user", "tool"})
    user, user_provenance = _mask_spans(messages, user_spans, user_placeholder, {"user"})
    return {"views": {"original": copy.deepcopy(list(messages)),
                      "observation_masked": observation, "user_masked": user},
            "mask_provenance": {"observation_masked": observation_provenance,
                                "user_masked": user_provenance},
            "intervention": "content_only_replacement",
            "message_structure_preserved": True}


def aggregate_token_logprobs(token_logprobs: Sequence[float]) -> dict:
    """Return the true joint score and a separately named normalized statistic."""
    values = [float(value) for value in token_logprobs]
    if not values or any(not math.isfinite(value) or value > 1e-6 for value in values):
        raise ProbeError("Token log probabilities must be finite, nonpositive and nonempty")
    joint = math.fsum(values)
    return {"joint_loglik": joint, "length_normalized_loglik": joint / len(values),
            "token_logprobs": values, "scored_token_count": len(values)}


def prediction_pairs(
    input_ids: Sequence[int], score_mask: Sequence[int], attention_mask: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    """Return (preceding-logit-position, target-token-id), excluding all padding.

    A causal LM's logits at j-1 predict input_ids[j]. Candidate tokens at j=0,
    or after a masked/padded predecessor, cannot be scored correctly.
    """
    attention = list(attention_mask) if attention_mask is not None else [1] * len(input_ids)
    if len(input_ids) != len(score_mask) or len(input_ids) != len(attention):
        raise ProbeError("IDs, attention and score masks must have equal length")
    if any(value not in (0, 1) for value in list(score_mask) + attention):
        raise ProbeError("Attention and score masks must be binary")
    pairs = []
    for position, selected in enumerate(score_mask):
        if selected:
            if position == 0 or not attention[position] or not attention[position - 1]:
                raise ProbeError("Candidate token has no unmasked causal predecessor")
            pairs.append((position - 1, int(input_ids[position])))
    if not pairs:
        raise ProbeError("No identifier tokens selected for scoring")
    return pairs


def sequence_loglik_from_logits(
    logits: Sequence[Sequence[float]], input_ids: Sequence[int], score_mask: Sequence[int],
    attention_mask: Sequence[int] | None = None,
) -> dict:
    """Standard-library reference scorer used to verify next-token alignment.

    This is intentionally small-input code; HFSequenceScorer uses Torch on only
    the selected positions rather than transferring full-vocabulary tensors.
    """
    if len(logits) != len(input_ids):
        raise ProbeError("A logit row is required for every input position")
    values = []
    for prediction_position, target in prediction_pairs(input_ids, score_mask, attention_mask):
        row = [float(value) for value in logits[prediction_position]]
        if not 0 <= target < len(row) or not row or any(not math.isfinite(v) for v in row):
            raise ProbeError("Logit row or target token is invalid")
        maximum = max(row)
        values.append(row[target] - maximum - math.log(math.fsum(math.exp(v - maximum) for v in row)))
    return aggregate_token_logprobs(values)


def dual_margins(view_scores: Mapping[str, Mapping[str, Any] | float], *, alpha: float = 1.0,
                 score_field: str = "length_normalized_loglik") -> dict:
    """Compute observation/user sensitivity and their proposed relative margin."""
    if not math.isfinite(alpha) or alpha < 0:
        raise ProbeError("alpha must be finite and nonnegative")
    scores = {}
    for name in VIEW_NAMES:
        if name not in view_scores:
            raise ProbeError(f"Missing view: {name}")
        entry = view_scores[name]
        value = entry[score_field] if isinstance(entry, Mapping) else entry
        scores[name] = float(value)
        if not math.isfinite(scores[name]):
            raise ProbeError("View scores must be finite")
    delta_obs = scores["original"] - scores["observation_masked"]
    delta_user = scores["original"] - scores["user_masked"]
    # At alpha=1 the original score cancels exactly; compute this direct form.
    risk = (scores["user_masked"] - scores["observation_masked"] if alpha == 1.0
            else delta_obs - alpha * delta_user)
    return {"delta_obs": delta_obs, "delta_user": delta_user, "risk": risk,
            "alpha": float(alpha), "score_field": score_field,
            "original_cancels": alpha == 1.0,
            "interpretation": "counterfactual_sensitivity_not_authorization"}


def gate_from_scores(view_scores, *, threshold: float, alpha: float = 1.0,
                     mode: str = "bilateral", high_risk: bool = True,
                     score_field: str = "length_normalized_loglik") -> dict:
    """Make an execution gate decision; no tool is executed by this module.

    All margins are retained for comparison. ``observation_only`` is a mechanism
    baseline, not a faithful reproduction of every original published defense.
    ``relative_alpha1`` exposes the algebraically equivalent two-view margin.
    """
    if mode not in {"bilateral", "observation_only", "relative_alpha1"}:
        raise ProbeError("Unknown gating mode")
    if not math.isfinite(threshold):
        raise ProbeError("Gate threshold must be finite")
    margins = dual_margins(view_scores, alpha=1.0 if mode == "relative_alpha1" else alpha,
                           score_field=score_field)
    value = margins["delta_obs"] if mode == "observation_only" else margins["risk"]
    block = bool(high_risk and value > threshold)
    return {**margins, "mode": mode, "threshold": float(threshold), "gate_score": value,
            "high_risk": bool(high_risk), "blocked": block,
            "decision": "block" if block else "allow",
            "reason": "risk_exempt" if not high_risk else "above_threshold" if block else "within_threshold",
            "arguments_checked": False, "actual_tool_execution": False}


def _encode(tokenizer, text):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    # Supports a native tokenizers.Tokenizer via an adapter, and HF tokenizers.
    return [int(value) for value in (encoded.ids if hasattr(encoded, "ids") else encoded)]


def prepare_sequence(tokenizer, context: str, tool_name: str, *, identifier_prefix: str = " ",
                     termination: str = "\n", score_termination: bool = True,
                     max_input_tokens: int | None = None) -> dict:
    """Jointly tokenize the entire fixed context+identifier, checking both edges.

    The formatting prefix (normally one space) is part of the serialized name
    event and its denominator. A terminating delimiter can optionally be scored
    to distinguish a complete name from a longer name with the same prefix.
    A boundary-crossing merge fails explicitly; this function does not splice
    independently tokenized strings or truncate the proposed identifier.
    """
    if not all(isinstance(value, str) for value in (context, tool_name, identifier_prefix, termination)):
        raise ProbeError("Context and serialized identifier parts must be strings")
    if not context or not tool_name or tool_name.strip() != tool_name:
        raise ProbeError("Context and exact, nonempty tool identifier are required")
    if score_termination and not termination:
        raise ProbeError("Scoring a termination requires a nonempty delimiter")
    prefix_ids = _encode(tokenizer, context)
    name_ids = _encode(tokenizer, context + identifier_prefix + tool_name)
    full_ids = _encode(tokenizer, context + identifier_prefix + tool_name + termination)
    if not prefix_ids or name_ids[:len(prefix_ids)] != prefix_ids or len(name_ids) <= len(prefix_ids):
        raise TokenBoundaryError("Prefix/identifier tokenization boundary is unstable")
    if full_ids[:len(name_ids)] != name_ids:
        raise TokenBoundaryError("Identifier/termination tokenization boundary is unstable")
    if max_input_tokens is not None and len(full_ids) > max_input_tokens:
        raise ProbeError(f"context_overflow:{len(full_ids)}>{max_input_tokens}; no truncation permitted")
    prefix_count = len(prefix_ids)
    name_count = len(name_ids) - prefix_count
    scored_end = len(full_ids) if score_termination else len(name_ids)
    mask = [int(prefix_count <= index < scored_end) for index in range(len(full_ids))]
    return {"input_ids": full_ids, "score_mask": mask,
            "input_token_count": prefix_count, "identifier_token_count": name_count,
            "termination_token_count": len(full_ids) - len(name_ids),
            "candidate_token_ids": full_ids[prefix_count:scored_end],
            "identifier_token_ids": name_ids[prefix_count:],
            "full_candidate_token_ids": full_ids[prefix_count:],
            "scored_token_count": scored_end - prefix_count,
            "input_sha256": hashlib.sha256(context.encode()).hexdigest(),
            "serialized_identifier": identifier_prefix + tool_name,
            "identifier_prefix": identifier_prefix, "termination": termination,
            "score_termination": bool(score_termination),
            "normalization_denominator": "serialized_identifier_and_termination_tokens" if score_termination
                                         else "serialized_identifier_tokens",
            "boundary_validation": "joint_encoding_prefix_and_identifier_exact"}


class HFSequenceScorer:
    """Teacher forcing for the same proposed tool in all counterfactual views.

    A forward call scores a batch of rows; this is independent of how many rows
    are scored. With no cached complete original score there are three fresh
    rows, irrespective of the original generation's first-token diagnostics.
    Right padding preserves absolute token positions; its cost is recorded.
    CUDA timing synchronizes every accelerator used by a dispatched model.
    """

    def __init__(self, model, tokenizer, *, batch_size: int = 3,
                 max_input_tokens: int | None = None):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ProbeError("batch_size must be a positive integer")
        if max_input_tokens is not None and max_input_tokens < 1:
            raise ProbeError("max_input_tokens must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self._scorer_id = uuid.uuid4().hex

    def _device(self, torch):
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, StopIteration):
            return torch.device(getattr(self.model, "device", "cpu"))

    def _cuda_devices(self, torch):
        devices = set()
        if torch.cuda.is_available():
            for value in getattr(self.model, "hf_device_map", {}).values():
                if isinstance(value, int):
                    devices.add(value)
                elif str(value).startswith("cuda"):
                    device = torch.device(value)
                    devices.add(device.index if device.index is not None else torch.cuda.current_device())
            device = self._device(torch)
            if device.type == "cuda":
                devices.add(device.index if device.index is not None else torch.cuda.current_device())
        return sorted(devices)

    def score(self, contexts: Mapping[str, str], tool_name: str, *, identifier_prefix: str = " ",
              termination: str = "\n", score_termination: bool = True,
              original_score: Mapping[str, Any] | None = None) -> dict:
        overall_started = time.perf_counter()
        if set(contexts) != set(VIEW_NAMES):
            raise ProbeError("Exactly original, observation_masked and user_masked contexts are required")
        rows = {name: prepare_sequence(self.tokenizer, contexts[name], tool_name,
                                      identifier_prefix=identifier_prefix, termination=termination,
                                      score_termination=score_termination,
                                      max_input_tokens=self.max_input_tokens) for name in VIEW_NAMES}
        proposed_ids = rows["original"]["full_candidate_token_ids"]
        if any(row["full_candidate_token_ids"] != proposed_ids for row in rows.values()):
            raise TokenBoundaryError("The same proposed identifier must have identical tokens in every view")
        views = {}
        if original_score is not None:
            expected = rows["original"]
            keys = ("input_sha256", "candidate_token_ids", "full_candidate_token_ids", "score_termination",
                    "normalization_denominator", "identifier_token_count", "scored_token_count")
            if (original_score.get("scorer_id") != self._scorer_id or
                any(original_score.get(key) != expected[key] for key in keys)):
                raise ProbeError("Cached original must be a complete score from this scorer and exact fixed event")
            aggregate = aggregate_token_logprobs(original_score.get("token_logprobs", []))
            if any(not math.isclose(float(original_score[key]), aggregate[key], abs_tol=1e-8)
                   for key in ("joint_loglik", "length_normalized_loglik")):
                raise ProbeError("Cached original aggregation is inconsistent")
            views["original"] = copy.deepcopy(dict(original_score))
        import torch

        devices = self._cuda_devices(torch)
        for device in devices:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        cost = {"model_forward_calls": 0, "scored_rows": 0, "input_tokens": 0,
                "candidate_tokens": 0, "scored_candidate_tokens": 0,
                "padded_model_input_tokens": 0, "batch_sizes": [],
                "reused_complete_original_score": original_score is not None,
                "autoreg_generation_tokens": 0, "synchronized_cuda_devices": devices,
                "preparation_wall_time_ms": (started - overall_started) * 1000}
        pending = [name for name in VIEW_NAMES if name not in views]
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise ProbeError("Tokenizer needs an existing pad or eos ID; vocabulary will not be modified")
        training = getattr(self.model, "training", False)
        self.model.eval()
        try:
            with torch.inference_mode():
                for offset in range(0, len(pending), self.batch_size):
                    names = pending[offset:offset + self.batch_size]
                    selected = [rows[name] for name in names]
                    width = max(len(row["input_ids"]) for row in selected)
                    ids = [row["input_ids"] + [pad_id] * (width - len(row["input_ids"])) for row in selected]
                    attention = [[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"]))
                                 for row in selected]
                    device = self._device(torch)
                    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
                    attention_mask = torch.tensor(attention, dtype=torch.long, device=device)
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                        use_cache=False, return_dict=True)
                    if output.logits.shape[:2] != input_ids.shape:
                        raise ProbeError("Model must return next-token logits for every input position")
                    for index, (name, row) in enumerate(zip(names, selected)):
                        pairs = prediction_pairs(row["input_ids"], row["score_mask"])
                        positions = torch.tensor([position for position, _ in pairs], device=output.logits.device)
                        targets = torch.tensor([target for _, target in pairs], device=output.logits.device)
                        selected_logits = output.logits[index].index_select(0, positions).float()
                        logprobs = torch.log_softmax(selected_logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
                        aggregate = aggregate_token_logprobs(logprobs.detach().cpu().tolist())
                        metadata = {key: value for key, value in row.items() if key not in {"input_ids", "score_mask"}}
                        views[name] = {**metadata, **aggregate, "scorer_id": self._scorer_id,
                                       "score_kind": "full_tool_identifier_sequence", "tool_name": tool_name}
                        cost["input_tokens"] += row["input_token_count"]
                        cost["candidate_tokens"] += len(row["full_candidate_token_ids"])
                        cost["scored_candidate_tokens"] += row["scored_token_count"]
                    cost["model_forward_calls"] += 1
                    cost["scored_rows"] += len(names)
                    cost["batch_sizes"].append(len(names))
                    cost["padded_model_input_tokens"] += len(names) * width
                    del output
            for device in devices:
                torch.cuda.synchronize(device)
        finally:
            if training:
                self.model.train()
        finished = time.perf_counter()
        cost["forward_scoring_wall_time_ms"] = (finished - started) * 1000
        cost["wall_time_ms"] = (finished - overall_started) * 1000
        cost["input_plus_candidate_tokens"] = cost["input_tokens"] + cost["candidate_tokens"]
        return {"tool_name": tool_name, "views": views, "cost": cost,
                "same_fixed_output": True, "same_candidate_token_ids": proposed_ids,
                "arguments_scored": False,
                "normalized_score_is_log_probability": False,
                "normalization_denominator": rows["original"]["normalization_denominator"],
                "scope": "identifier_choice_only"}
