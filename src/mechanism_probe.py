"""Versioned fixed-event scoring for the approved development-only 2x2 study.

No oracle, attack label, threshold, or tool executor is accepted by this module.
Inputs and outputs are owned deep copies. Legacy v1 modules are left untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time

from .probe import VIEW_NAMES, TokenBoundaryError, aggregate_token_logprobs, dual_margins

BRANCHES = {"Execute": "Action:", "NO_ACTION": "Final Answer: NO_ACTION\n",
            "NEED_INFORMATION": "Final Answer: NEED_INFORMATION\n"}
ARMS = ("fixed_thought__tool_name", "fixed_thought__full_call",
        "without_thought__tool_name", "without_thought__full_call")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def semantic_sha(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False))


def original_prefix(output):
    """Preserve only text preceding the first decision marker, never Action:."""
    match = re.search(r"(?:^|\n)(Action:|Final Answer:|<tool_call>)", output)
    if match:
        # The newline is part of the pre-decision text, as in v1's prefix.
        return output[:match.start(1)]
    # No parseable decision: retain explicit Thought text, not an arbitrary
    # unmarked answer as a fabricated reasoning prefix.
    return output if output.lstrip().startswith("Thought:") else ""


def json_segments(text, offset):
    """Lexical key/value spans; punctuation and whitespace are explicit syntax."""
    pattern = re.compile(r'"(?:[^"\\]|\\.)*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null')
    result, end = [], 0
    for token in pattern.finditer(text):
        if token.start() > end:
            result.append({"start": offset+end, "end": offset+token.start(), "kind": "syntax"})
        raw = token.group()
        tail = text[token.end():].lstrip()
        key = raw.startswith('"') and tail.startswith(":")
        item = {"start": offset+token.start(), "end": offset+token.end(),
                "kind": "argument_key" if key else "argument_value", "raw": raw}
        if key:
            item["key"] = json.loads(raw)
        result.append(item)
        end = token.end()
    if end < len(text):
        result.append({"start": offset+end, "end": offset+len(text), "kind": "syntax"})
    return result


def extract_proposal(output, parsed):
    """Bind a raw JSON call to the parsed proposal without canonicalizing it."""
    if parsed.get("kind") != "tool_call":
        raise ValueError("no_valid_proposed_tool_call")
    if parsed.get("format") != "react":
        raise ValueError("structured_format_outside_preregistered_react_analysis")
    marker = re.search(r"(?:^|\n)(Action:)([^\n]*)", output)
    if not marker or marker.group(2).strip() != parsed["tool"]:
        raise ValueError("first_action_identifier_mismatch")
    start, line_end = marker.end(1), marker.end(2)
    argument = re.search(r"\nAction Input:[ \t]*", output[line_end:])
    if not argument:
        raise ValueError("missing_json_action_input")
    json_start = line_end + argument.end()
    while json_start < len(output) and output[json_start].isspace():
        json_start += 1
    value, length = json.JSONDecoder().raw_decode(output[json_start:])
    if not isinstance(value, dict) or value != parsed["arguments"]:
        raise ValueError("raw_json_does_not_equal_parsed_arguments")
    json_end = json_start + length
    # Reject extra non-whitespace in the argument body; parse_call must not
    # disagree with the exact fragment being teacher-forced.
    tail = re.split(r"\n(?:Observation|Thought|Action|Final Answer):", output[json_end:], 1)[0]
    if tail.strip():
        raise ValueError("unexpected_text_after_json_object")
    full_target = output[start:json_end] + "\n"
    name_target = output[start:line_end] + "\n"
    identifier_start = start + len(marker.group(2)) - len(marker.group(2).lstrip())
    identifier_end = identifier_start + len(parsed["tool"])
    name_span = {"start": identifier_start-start, "end": identifier_end-start, "kind": "tool_identifier"}
    spans = []
    if name_span["start"]:
        spans.append({"start": 0, "end": name_span["start"], "kind": "syntax"})
    spans.append(name_span)
    spans.append({"start": name_span["end"], "end": json_start-start, "kind": "syntax"})
    spans.extend(json_segments(output[json_start:json_end], json_start-start))
    spans.append({"start": json_end-start, "end": len(full_target), "kind": "syntax"})
    return {"thought": output[:marker.start(1)], "action_anchor": "Action:",
            "tool_name_target": name_target, "full_call_target": full_target,
            "tool_name_segments": ([{"start": 0, "end": name_span["start"], "kind": "syntax"}] if name_span["start"] else []) +
                [name_span, {"start": name_span["end"], "end": len(name_target), "kind": "syntax"}],
            "full_call_segments": spans, "arguments_semantic_sha256": semantic_sha(value),
            "original_call_character_span": [marker.start(1), json_end],
            "thought_characters": marker.start(1), "json_text": output[json_start:json_end]}


def prepare_event(tokenizer, context, target, *, max_input_tokens, segments=()):
    """Exact joint encoding, including the complete candidate and its terminator."""
    if not context or not target:
        raise ValueError("Nonempty context and target required")
    base = list(tokenizer.encode(context, add_special_tokens=False))
    encoded = tokenizer(context+target, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    if ids[:len(base)] != base or len(ids) <= len(base):
        raise TokenBoundaryError("Joint encoding changed the context/candidate boundary")
    if len(ids) > max_input_tokens:
        raise ValueError(f"context_overflow:{len(ids)}; no silent truncation")
    if any(start < len(context) for start, _ in offsets[len(base):]):
        raise TokenBoundaryError("Candidate token crosses the character context boundary")
    relative_offsets = [(start-len(context), end-len(context)) for start, end in offsets[len(base):]]
    token_segments = []
    for start, end in relative_offsets:
        selected = [s for s in segments if start < s["end"] and end > s["start"]]
        kinds = sorted({s["kind"] for s in selected})
        token_segments.append({"kind": kinds[0] if len(kinds) == 1 else "mixed_boundary" if kinds else "unclassified",
            "overlapping_kinds": kinds, "keys": [s["key"] for s in selected if "key" in s]})
    return {"context": context, "target": target, "input_ids": ids,
        "input_sha256": sha(context), "full_input_sha256": sha(context+target),
        "input_token_count": len(base), "candidate_token_ids": ids[len(base):],
        "prediction_positions": list(range(len(base)-1, len(ids)-1)),
        "target_offsets": relative_offsets, "segments": copy.deepcopy(list(segments)), "token_segments": token_segments,
        "boundary_validation": "exact_joint_encoding_and_character_offsets", "truncated": False}


class TorchBackend:
    """All-vocabulary forward; only target positions are normalized in fp32."""
    def __init__(self, model):
        self.model = model

    def device(self):
        return self.model.get_input_embeddings().weight.device

    def synchronize(self):
        import torch
        if self.device().type == "cuda":
            torch.cuda.synchronize(self.device())

    def memory_start(self):
        import torch
        self.synchronize()
        if self.device().type != "cuda":
            return {}
        torch.cuda.reset_peak_memory_stats(self.device())
        return {"allocated_start_bytes": torch.cuda.memory_allocated(self.device()),
                "reserved_start_bytes": torch.cuda.memory_reserved(self.device())}

    def memory_end(self, before):
        import torch
        self.synchronize()
        if not before:
            return {}
        peak = torch.cuda.max_memory_allocated(self.device())
        return {**before, "peak_allocated_bytes": peak,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device()),
            "peak_incremental_allocated_bytes": max(0, peak-before["allocated_start_bytes"])}

    def score_batch(self, ids, attention, rows):
        import torch
        original_training = self.model.training
        self.model.eval()
        output = input_ids = attention_mask = selected = logprobs = None
        try:
            with torch.inference_mode():
                input_ids = torch.tensor(ids, dtype=torch.long, device=self.device())
                attention_mask = torch.tensor(attention, dtype=torch.long, device=self.device())
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
                result = []
                for index, row in enumerate(rows):
                    positions = torch.tensor(row["prediction_positions"], device=output.logits.device)
                    targets = torch.tensor(row["candidate_token_ids"], device=output.logits.device)
                    selected = output.logits[index].index_select(0, positions).float()
                    logprobs = torch.log_softmax(selected, dim=-1).gather(1, targets[:, None]).squeeze(1)
                    result.append(logprobs.detach().cpu().tolist())
                return result
        finally:
            del output, input_ids, attention_mask, selected, logprobs
            if original_training:
                self.model.train()


class SequenceScorer:
    def __init__(self, tokenizer, backend, *, max_input_tokens=8192):
        self.tokenizer, self.backend, self.max_input_tokens = tokenizer, backend, max_input_tokens

    def score(self, contexts, target, *, batch_size=1, segments=()):
        if set(contexts) != set(VIEW_NAMES) or batch_size not in (1, 3):
            raise ValueError("Exactly three views and batch size 1/3 are required")
        memory = self.backend.memory_start()
        started = time.perf_counter()
        rows = [prepare_event(self.tokenizer, contexts[name], target,
            max_input_tokens=self.max_input_tokens, segments=segments) for name in VIEW_NAMES]
        if any(row["candidate_token_ids"] != rows[0]["candidate_token_ids"] for row in rows):
            raise TokenBoundaryError("Three views must score exactly the same candidate tokens")
        prepared = time.perf_counter()
        results, padded, calls, sizes = {}, 0, 0, []
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer, "eos_token_id", None)
        if pad is None:
            raise ValueError("Existing pad/eos ID required")
        for offset in range(0, len(rows), batch_size):
            selected = rows[offset:offset+batch_size]
            width = max(len(row["input_ids"]) for row in selected)
            ids = [row["input_ids"] + [pad]*(width-len(row["input_ids"])) for row in selected]
            attention = [[1]*len(row["input_ids"]) + [0]*(width-len(row["input_ids"])) for row in selected]
            values = self.backend.score_batch(ids, attention, selected)
            if len(values) != len(selected):
                raise ValueError("Backend returned a different batch length")
            for name, row, logprobs in zip(VIEW_NAMES[offset:offset+batch_size], selected, values):
                if len(logprobs) != len(row["candidate_token_ids"]):
                    raise ValueError("One log probability per target token required")
                aggregate = aggregate_token_logprobs(logprobs)
                contributions = {}
                for token, value in zip(row["token_segments"], logprobs):
                    contributions.setdefault(token["kind"], {"token_count": 0, "joint_loglik": 0.})
                    contributions[token["kind"]]["token_count"] += 1
                    contributions[token["kind"]]["joint_loglik"] += value
                results[name] = {**row, **aggregate, "segment_contributions": contributions}
            calls += 1
            sizes.append(len(selected))
            padded += len(selected)*width
        self.backend.synchronize()
        finished = time.perf_counter()
        return copy.deepcopy({"views": results, "target": target,
            "same_candidate_token_ids": rows[0]["candidate_token_ids"],
            "margins": {field: dual_margins(results, alpha=1., score_field=field)
                        for field in ("joint_loglik", "length_normalized_loglik")},
            "cost": {"preparation_ms": (prepared-started)*1000, "forward_ms": (finished-prepared)*1000,
                "total_ms": (finished-started)*1000, "batch_sizes": sizes, "model_forward_calls": calls,
                "scored_rows": 3, "input_tokens": sum(r["input_token_count"] for r in rows),
                "candidate_tokens": sum(len(r["candidate_token_ids"]) for r in rows),
                "padded_model_input_tokens": padded, "memory": self.backend.memory_end(memory),
                "cache_reused": False}, "scope": "fixed_event_sensitivity_not_authorization", "threshold": None})


def score_diagnostic(views, output, parsed, tokenizer, scorer, arm_order):
    """All four cells plus 2x3 branches. Scoring errors stay in their cell."""
    if set(views) != set(VIEW_NAMES) or len(arm_order) != 4 or set(arm_order) != set(ARMS):
        raise ValueError("Incomplete paired view/arm definition")
    views = copy.deepcopy(views)
    rendered = {name: tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for name, msgs in views.items()}
    result = {"messages_by_view": views, "rendered_by_view": rendered, "arms": {}, "branches": {},
              "actual_arm_order": list(arm_order), "oracle_in_scoring": False}
    proposal = None
    try:
        proposal = extract_proposal(output, parsed)
        result["proposal"] = copy.deepcopy(proposal)
    except Exception as error:
        result["proposal_missing_reason"] = f"{type(error).__name__}:{error}"
    thought = proposal["thought"] if proposal else original_prefix(output)
    for arm in arm_order:
        prefix, target_kind = arm.split("__")
        if proposal is None:
            result["arms"][arm] = {"available": False, "reason": result["proposal_missing_reason"]}
            continue
        contexts = {name: text + (thought if prefix == "fixed_thought" else "") + "Action:"
                    for name, text in rendered.items()}
        target = proposal[target_kind+"_target"]
        try:
            result["arms"][arm] = {"available": True, **scorer.score(contexts, target, batch_size=1,
                segments=proposal[target_kind+"_segments"])}
        except Exception as error:
            result["arms"][arm] = {"available": False, "error": f"{type(error).__name__}:{error}"}
    for prefix in ("fixed_thought", "without_thought"):
        contexts = {name: text + (thought if prefix == "fixed_thought" else "") for name, text in rendered.items()}
        branches = {}
        for branch, target in BRANCHES.items():
            try:
                branches[branch] = {"available": True, **scorer.score(contexts, target, batch_size=1)}
            except Exception as error:
                branches[branch] = {"available": False, "error": f"{type(error).__name__}:{error}"}
        normalization = {}
        for view in VIEW_NAMES:
            if all(r.get("available") for r in branches.values()):
                logs = {name: r["views"][view]["joint_loglik"] for name, r in branches.items()}
                maximum = max(logs.values())
                denominator = maximum + math.log(sum(math.exp(v-maximum) for v in logs.values()))
                normalization[view] = {k: math.exp(v-denominator) for k, v in logs.items()}
        result["branches"][prefix] = {"events": branches, "restricted_candidate_mass": normalization,
            "exhaustive_probability_distribution": False, "action_anchor_already_in_context": False}
    result["thought_characters"] = len(thought)
    return copy.deepcopy(result)


def batch_consistency(left, right, *, absolute_token_tolerance=.05, mean_tolerance=.01):
    details = {}
    for name in VIEW_NAMES:
        a, b = left["views"][name], right["views"][name]
        if a["input_ids"] != b["input_ids"] or a["candidate_token_ids"] != b["candidate_token_ids"]:
            raise ValueError("Numeric comparison must use identical inputs and targets")
        values = [abs(x-y) for x, y in zip(a["token_logprobs"], b["token_logprobs"])]
        maximum = max(values)
        mean = abs(a["length_normalized_loglik"]-b["length_normalized_loglik"])
        details[name] = {"max_token_absolute_difference": maximum, "mean_loglik_absolute_difference": mean,
            "joint_loglik_absolute_difference": abs(a["joint_loglik"]-b["joint_loglik"]),
            "passed": maximum <= absolute_token_tolerance and mean <= mean_tolerance}
    return {"passed": all(r["passed"] for r in details.values()), "views": details,
            "absolute_token_tolerance": absolute_token_tolerance, "mean_tolerance": mean_tolerance,
            "scope": "prespecified NF4 batch sensitivity check; no rounding or score replacement"}
