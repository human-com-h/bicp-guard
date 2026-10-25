"""Translate recorded ToolBench history to the same ReAct prompt family."""
import json


def tool_specs(sample):
    tools = {}
    for function in sample["functions"]:
        if function["name"] == "Finish":
            continue
        schema = function.get("parameters", {})
        required = schema.get("required", [])
        parameters = [dict(name=name, type=value.get("type", "string"),
                           description=value.get("description", ""), required=name in required)
                      for name, value in schema.get("properties", {}).items()]
        tools[function["name"]] = {"name": function["name"], "summary": function.get("description", ""),
                                    "parameters": parameters, "returns": []}
    return tools


def build_replay_prompt(sample, prompts):
    tools = tool_specs(sample)
    history = []
    inputs = []
    for message in sample["messages_before_target"]:
        if message["role"] == "user":
            inputs.append(message.get("content", ""))
        elif message["role"] == "assistant":
            if message.get("content"):
                history.append("Thought: "+message["content"])
            if message.get("function_call"):
                call = message["function_call"]
                history.append(f"Action: {call['name']}\nAction Input: {call['arguments']}")
        elif message["role"] == "function":
            history.append("Observation: "+message.get("content", ""))
    system, user = prompts.PROMPT_DICT["InjecAgent"]
    rendered = user.format(available_tool_descriptions=json.dumps(list(tools.values()), indent=True),
                           input="\n".join(inputs), available_tool_names=list(tools),
                           agent_scratchpad="\n"+"\n".join(history)+"\n")
    return [{"role": "system", "content": system}, {"role": "user", "content": rendered}], tools
