"""Pinned official tool runtime without importing tasks, ground truth or utility."""
import ast,copy,importlib,sys,types
from pathlib import Path

def open_public_workspace(root):
    from .run_agentdojo_bicp import verify_source
    verify_source(root)
    source=Path(root)/'data/upstream/AgentDojo/src';package=source/'agentdojo'
    if 'agentdojo' in sys.modules:raise RuntimeError('Public native loader requires a fresh process')
    sys.path.insert(0,str(source))
    importlib.import_module('agentdojo')
    for name,relative in [('agentdojo.default_suites','default_suites'),('agentdojo.default_suites.v1','default_suites/v1'),
                          ('agentdojo.default_suites.v1.tools','default_suites/v1/tools'),('agentdojo.default_suites.v1.workspace','default_suites/v1/workspace')]:
        mod=types.ModuleType(name);mod.__path__=[str(package/relative)];mod.__package__=name
        sys.modules[name]=mod
    rm=importlib.import_module('agentdojo.functions_runtime')
    path=package/'default_suites/v1/workspace/task_suite.py'
    nodes=[]
    for n in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(n,ast.ImportFrom) and (n.module.startswith('agentdojo.default_suites.v1.tools.') or n.module=='agentdojo.functions_runtime'):nodes.append(n)
        if isinstance(n,ast.ClassDef) and n.name=='WorkspaceEnvironment':nodes.append(n)
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TOOLS' for t in n.targets):nodes.append(n)
    ns={'__name__':'public_workspace_definition'}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    functions=rm.FunctionsRuntime([rm.make_function(f) for f in ns['TOOLS']])
    # Execute only the public formatter definition, not its pipeline imports.
    import yaml
    from pydantic import BaseModel
    from collections.abc import Callable
    fpath=package/'agent_pipeline/tool_execution.py'
    node=next(n for n in ast.parse(fpath.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='tool_result_to_str')
    fmt={'yaml':yaml,'BaseModel':BaseModel,'Callable':Callable,'FunctionReturnType':rm.FunctionReturnType}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(fpath),'exec'),fmt)
    return ns['WorkspaceEnvironment'],functions,rm,fmt['tool_result_to_str']

def environment_from_public(environment_type,payload):
    env=environment_type.model_validate(copy.deepcopy(payload['initial_state']))
    if env.model_dump(mode='json')!=payload['initial_state']:raise ValueError('Public clean-state round trip changed')
    return env
