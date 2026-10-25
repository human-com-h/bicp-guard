"""Paired, family-preserving descriptive analysis; no population guarantee."""
from collections import defaultdict
import math
import random


SHARED_PROPOSAL = {"no_defense", "observation_only", "bilateral", "relative_alpha1", "masked_regeneration_shadow"}


def outcome_bounds(record, method, field):
    item = record.get("methods", {}).get(method, {})
    if not item.get("available"):
        return 0, 1
    if field == "attack_success" and record["source"] == "injecagent" and item.get("unresolved_stage2"):
        return 0, 1
    value = item.get(field)
    if not isinstance(value, bool):
        raise ValueError("Available outcomes must have explicit boolean labels")
    return int(value), int(value)


def paired_bounds(record, method, reference, field):
    new, old = outcome_bounds(record, method, field), outcome_bounds(record, reference, field)
    common_unknown = False
    if field == "attack_success" and record["source"] == "injecagent" and method in SHARED_PROPOSAL and reference in SHARED_PROPOSAL:
        a, b = record.get("methods", {}).get(method, {}), record.get("methods", {}).get(reference, {})
        common_unknown = (a.get("available") and b.get("available") and a.get("unresolved_stage2") and b.get("unresolved_stage2")
            and a.get("official_stage1") == b.get("official_stage1") and a.get("official_stage2") == b.get("official_stage2"))
    if common_unknown:
        return 0, 0, True
    return new[0] - old[1], new[1] - old[0], False


def quantile(values, probability):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] * (hi - index) + values[hi] * (index - lo) if lo != hi else values[lo]


def family_sensitivity(group_totals, *, resampling_seed=42, repeats=10000):
    """Resample entire observed groups; intervals describe composition sensitivity.

    Each tuple is (lower numerator, upper numerator, denominator). Missing
    outcomes retain bounds; no-opportunity groups retain a zero denominator.
    A zero observed interval cannot certify absence of unseen failures.
    """
    groups = [group_totals[group] for group in sorted(group_totals)]
    if not groups or repeats < 1:
        raise ValueError("At least one group and positive repeat count required")
    if any(n < 0 or lo > hi or not all(math.isfinite(v) for v in (lo, hi, n)) for lo, hi, n in groups):
        raise ValueError("Invalid grouped counts or bounds")
    rng = random.Random(resampling_seed)
    lower, upper, undefined = [], [], 0
    for _ in range(repeats):
        chosen = [groups[rng.randrange(len(groups))] for _ in groups]
        denominator = sum(row[2] for row in chosen)
        if not denominator:
            undefined += 1
            continue
        lower.append(sum(row[0] for row in chosen) / denominator)
        upper.append(sum(row[1] for row in chosen) / denominator)
    return {"groups": len(groups), "repeats": repeats, "undefined_replicates": undefined,
        "resampling_seed": resampling_seed, "sensitivity_95": [quantile(lower, .025), quantile(upper, .975)],
        "scope": "observed-family composition sensitivity, not a population CI or proof of independence"}


def grouped_outcome(records, method, field, *, resampling_seed=42, repeats=10000):
    groups = defaultdict(lambda: [0, 0, 0])
    unknown = 0
    for record in records:
        lo, hi = outcome_bounds(record, method, field)
        unknown += lo != hi
        totals = groups[record["group_id"]]
        for index, value in enumerate((lo, hi, 1)):
            totals[index] += value
    if not records:
        return None
    return {"records": len(records), "unknown": unknown,
        "rate_bounds": [sum(row[0] for row in groups.values()) / len(records),
                        sum(row[1] for row in groups.values()) / len(records)],
        "family_counts": dict(groups), "resampling": family_sensitivity(groups, resampling_seed=resampling_seed, repeats=repeats)}


def grouped_difference(records, method, reference, field, *, resampling_seed=42, repeats=10000):
    groups = defaultdict(lambda: [0, 0, 0])
    common_unknown, discordant_new_only, discordant_old_only, unknown_pairs = 0, 0, 0, 0
    for record in records:
        lo, hi, common = paired_bounds(record, method, reference, field)
        common_unknown += common
        unknown_pairs += lo != hi
        a, b = outcome_bounds(record, method, field), outcome_bounds(record, reference, field)
        if a[0] == a[1] and b[0] == b[1]:
            discordant_new_only += a[0] == 1 and b[0] == 0
            discordant_old_only += a[0] == 0 and b[0] == 1
        for index, value in enumerate((lo, hi, 1)):
            groups[record["group_id"]][index] += value
    if not records:
        return None
    return {"method": method, "reference": reference, "field": field, "records": len(records),
        "difference_bounds": [sum(row[0] for row in groups.values()) / len(records),
                              sum(row[1] for row in groups.values()) / len(records)],
        "common_unresolved_shared_outcomes": common_unknown, "unknown_pairs": unknown_pairs,
        "known_discordant_method_only": discordant_new_only, "known_discordant_reference_only": discordant_old_only,
        "family_counts": dict(groups), "resampling": family_sensitivity(groups, resampling_seed=resampling_seed, repeats=repeats),
        "direction": "method minus reference; ASR negative is favorable, utility positive is favorable"}
