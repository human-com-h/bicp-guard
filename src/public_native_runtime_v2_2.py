"""Pinned tools/environment definitions only; no native task or utility imports."""
import ast, importlib, sys, types
from pathlib import Path

def open_public_native(root):
    from .run_agentdojo_bicp import verify_source
    verify_source(root)
    source=Path(root)/'data/upstream/AgentDojo/src';package=source/'agentdojo'
    if 'agentdojo' in sys.modules:raise RuntimeError('Public runtime requires a fresh process')
    sys.path.insert(0,str(source));importlib.import_module('agentdojo')
    for suffix in ('default_suites','default_suites.v1','default_suites.v1.tools',*[f'default_suites.v1.{s}' for s in ('workspace','slack','travel','banking')]):
        name='agentdojo.'+suffix;mod=types.ModuleType(name);mod.__path__=[str(package/Path(*suffix.split('.')))];mod.__package__=name;sys.modules[name]=mod
    rm=importlib.import_module('agentdojo.functions_runtime')
    import yaml
    from pydantic import BaseModel
    from collections.abc import Callable
    path=package/'agent_pipeline/tool_execution.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='tool_result_to_str')
    ns={'yaml':yaml,'BaseModel':BaseModel,'Callable':Callable,'FunctionReturnType':rm.FunctionReturnType}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns);formatter=ns['tool_result_to_str'];parts={}
    for suite in ('workspace','slack','travel','banking'):
        path=package/f'default_suites/v1/{suite}/task_suite.py';nodes=[]
        for n in ast.parse(path.read_text(encoding='utf-8')).body:
            if isinstance(n,ast.ImportFrom) and n.module and (n.module.startswith('agentdojo.default_suites.v1.tools.') or n.module=='agentdojo.functions_runtime'):nodes.append(n)
            if isinstance(n,ast.ClassDef) and n.name.endswith('Environment'):nodes.append(n)
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TOOLS' for t in n.targets):nodes.append(n)
        ns={'__name__':'public_'+suite+'_definition'};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        env=next(v for k,v in ns.items() if k.endswith('Environment') and k!='TaskEnvironment')
        parts[suite]=(env,rm.FunctionsRuntime([rm.make_function(f) for f in ns['TOOLS']]),rm,formatter)
    return parts
