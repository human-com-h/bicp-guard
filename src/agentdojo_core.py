"""Load unmodified AgentDojo core without eagerly importing unused providers.

This compatibility loader supplies the documented package re-exports explicitly.
It changes import initialization, not tasks, tools, runtime, or utility checks.
"""
import importlib
import ast
from pathlib import Path
import sys
import types


def load_core(checkout: Path, extra_packages: Path | None = None):
    if extra_packages:
        sys.path.insert(0, str(extra_packages))
    source = checkout/"src"
    if not (source/"agentdojo/functions_runtime.py").exists():
        raise FileNotFoundError("Pinned AgentDojo checkout missing")
    if "agentdojo" in sys.modules:
        raise RuntimeError("Use a fresh process so import initialization is explicit")
    sys.path.insert(0, str(source))
    importlib.import_module("agentdojo")
    for name, directory in (("agentdojo.agent_pipeline", source/"agentdojo/agent_pipeline"),
                            ("agentdojo.task_suite", source/"agentdojo/task_suite")):
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        package.__spec__ = importlib.util.spec_from_file_location(name, directory/"__init__.py", submodule_search_locations=[str(directory)])
        sys.modules[name] = package
        setattr(sys.modules["agentdojo"], name.rsplit(".", 1)[1], package)
    # Tool serialization imports one Google-provider sentinel even when no
    # Google model is used. Extract its literal from the pinned source rather
    # than importing an unused SDK or changing official tool execution code.
    google_path = source/"agentdojo/agent_pipeline/llms/google_llm.py"
    google = types.ModuleType("agentdojo.agent_pipeline.llms.google_llm")
    for node in ast.parse(google_path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EMPTY_FUNCTION_NAME" for t in node.targets):
            google.EMPTY_FUNCTION_NAME = ast.literal_eval(node.value)
    if not hasattr(google, "EMPTY_FUNCTION_NAME"):
        raise ValueError("Pinned Google sentinel changed")
    sys.modules[google.__name__] = google
    tasks = importlib.import_module("agentdojo.base_tasks")
    suite_type = importlib.import_module("agentdojo.task_suite.task_suite").TaskSuite
    package = sys.modules["agentdojo.task_suite"]
    for name in ("BaseUserTask", "BaseInjectionTask", "TaskDifficulty"):
        setattr(package, name, getattr(tasks, name))
    package.TaskSuite = suite_type
    suites = importlib.import_module("agentdojo.task_suite.load_suites")
    for name in ("get_suite", "get_suites", "register_suite"):
        setattr(package, name, getattr(suites, name))
    ground_truth = importlib.import_module("agentdojo.agent_pipeline.ground_truth_pipeline").GroundTruthPipeline
    runtime = importlib.import_module("agentdojo.functions_runtime")
    return suites, ground_truth, runtime
