"""Parse proposed calls as data. No external tools are executed."""
import ast
import json
import re


def parse_arguments(text):
    try:
        value = json.loads(text)
        mode = "json"
    except (json.JSONDecodeError, TypeError):
        try:
            value = ast.literal_eval(text)
            mode = "python_literal"
        except (ValueError, SyntaxError, TypeError):
            return None, "invalid"
    return (value, mode) if isinstance(value, dict) else (None, "not_object")


def parse_call(output, candidates):
    if not isinstance(output, str) or not output.strip():
        return {"kind": "invalid", "reason": "empty_output"}
    action = re.search(r"(?:^|\n)Action:\s*([^\n]+)", output)
    final = re.search(r"(?:^|\n)Final Answer:", output)
    if final and (not action or final.start() < action.start()):
        return {"kind": "final_answer", "text": output[final.end():].strip()}
    if not action:
        # Native structured formats are supported for diagnostics as well.
        structured = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", output, re.S)
        if structured:
            try:
                payload = json.loads(structured.group(1))
                name, arguments = payload["name"], payload["arguments"]
                if name in candidates and isinstance(arguments, dict):
                    return {"kind": "tool_call", "tool": name, "arguments": arguments,
                            "argument_format": "json", "format": "structured"}
            except (KeyError, ValueError, TypeError):
                pass
        return {"kind": "invalid", "reason": "no_action_or_final_answer"}
    name = action.group(1).strip()
    if name not in candidates:
        return {"kind": "invalid", "reason": "unknown_tool", "tool": name}
    tail = output[action.end():]
    args = re.search(r"(?:^|\n)Action Input:\s*", tail)
    if not args:
        return {"kind": "invalid", "reason": "missing_arguments", "tool": name}
    raw = re.split(r"\n(?:Observation|Thought|Action|Final Answer):", tail[args.end():], 1)[0].strip()
    arguments, mode = parse_arguments(raw)
    if arguments is None:
        return {"kind": "invalid", "reason": "invalid_arguments", "tool": name,
                "raw_arguments": raw}
    return {"kind": "tool_call", "tool": name, "arguments": arguments,
            "argument_format": mode, "format": "react"}


def validate_schema(arguments, tool):
    """Basic independent diagnostic; it never changes the undefended decision."""
    issues = []
    types = {"string": str, "integer": int, "number": (int, float),
             "boolean": bool, "array": list, "object": dict}
    parameters = {p["name"]: p for p in tool.get("parameters", [])}
    for name, parameter in parameters.items():
        if parameter.get("required") and name not in arguments:
            issues.append(f"missing:{name}")
        if name in arguments and parameter.get("type") in types:
            value = arguments[name]
            expected = types[parameter["type"]]
            if not isinstance(value, expected) or (parameter["type"] in {"number", "integer"} and isinstance(value, bool)):
                issues.append(f"type:{name}")
    issues.extend(f"extra:{name}" for name in arguments if name not in parameters)
    return {"valid": not issues, "issues": issues, "coverage": "required_and_top_level_types_only"}
