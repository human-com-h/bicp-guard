"""Pinned InjecAgent prompts, task-level partitions and cached simulation."""
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from .parsing import parse_arguments


GROUP_FIELDS = ("User Instruction", "User Tool", "Tool Parameters", "Tool Response Template")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def group_id(case):
    return digest({key: case[key] for key in GROUP_FIELDS})[:16]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tools(upstream):
    tools = {}
    for toolkit in json.loads((Path(upstream) / "data/tools.json").read_text(encoding="utf-8")):
        for tool in toolkit["tools"]:
            name = toolkit["toolkit"] + tool["name"]
            tools[name] = dict(tool, name=name)
    return tools


def load_cases(upstream, setting="base"):
    rows = []
    for attack in ("dh", "ds"):
        path = Path(upstream) / f"data/test_cases_{attack}_{setting}.json"
        for index, original in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            case = dict(original, case_id=f"{setting}:{attack}:{index:04d}", attack=attack,
                        group_id=group_id(original))
            rows.append(case)
    return rows


def build_splits(cases, seed):
    groups = sorted({case["group_id"] for case in cases})
    random.Random(seed).shuffle(groups)
    # 3/3/11 groups at the pinned 17-task version. All related variants share a split.
    n_dev = max(1, round(len(groups) * 0.18))
    n_cal = max(1, round(len(groups) * 0.18))
    assignment = {group: ("development" if i < n_dev else "calibration" if i < n_dev + n_cal else "test")
                  for i, group in enumerate(groups)}
    return {"seed": seed, "group_fields": list(GROUP_FIELDS), "group_assignment": assignment,
            "samples": [{"case_id": c["case_id"], "group_id": c["group_id"], "split": assignment[c["group_id"]]}
                        for c in cases]}


def clean_controls(cases):
    unique = {}
    for case in cases:
        if case["group_id"] not in unique:
            clean = dict(case)
            clean["case_id"] = "clean:" + case["group_id"]
            clean["attack"] = "clean_diagnostic"
            clean["Tool Response"] = '"' + case["Tool Response Template"].replace(
                "<Attacker Instruction>", "No additional instructions are present in this content.") + '"'
            clean["Attacker Instruction"] = ""
            clean["control_provenance"] = "synthetic_neutral_replacement"
            unique[case["group_id"]] = clean
    return list(unique.values())


def build_prompt(case, tools, prompt_module, stage2=None):
    system, user = prompt_module.PROMPT_DICT["InjecAgent"]
    candidates = [case["User Tool"]] + case["Attacker Tools"]
    history = (f"\n{case['Thought']}\nAction: {case['User Tool']}\n"
               f"Action Input: {case['Tool Parameters']}\nObservation: {case['Tool Response']}\n")
    if stage2:
        history += (f"Thought: {stage2['thought']}\nAction: {case['Attacker Tools'][0]}\n"
                    f"Action Input: {stage2['parameters']}\nObservation: {stage2['response']}\n")
    user_text = user.format(available_tool_descriptions=json.dumps([tools[t] for t in candidates], indent=True),
                            input=case["User Instruction"], available_tool_names=candidates,
                            agent_scratchpad=history)
    return [{"role": "system", "content": system}, {"role": "user", "content": user_text}]


class CachedSimulator:
    def __init__(self, upstream):
        self.responses = json.loads((Path(upstream) / "data/attacker_simulated_responses.json").read_text())
        self.canonical = {}
        for key, response in self.responses.items():
            tool, parameters = key[1:-1].split(",", 1)
            parsed, mode = parse_arguments(parameters)
            if parsed is not None:
                self.canonical.setdefault((tool, digest(parsed)), response)

    def get(self, tool, parameters):
        key = f"({tool},{parameters})"
        if key in self.responses:
            return self.responses[key], "official_cache_exact"
        parsed, mode = parse_arguments(parameters)
        if parsed is not None and (tool, digest(parsed)) in self.canonical:
            return self.canonical[(tool, digest(parsed))], "official_cache_canonical_parameters"
        return None, "cache_miss_unresolved"
