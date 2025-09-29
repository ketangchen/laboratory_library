import os
import pathlib
from pathlib import Path

def experiment_home_path():
    return pathlib.Path(__file__).parent.resolve()

def root_fuzz_result_path():
    return os.path.join(experiment_home_path(), "_fuzz_result")

def root_asan_result_path():
    return os.path.join(experiment_home_path(), "_asan_result")

def root_gcov_result_path():
    return os.path.join(experiment_home_path(), "_gcov_result")

def container_home_path():
    return "/root"

def container_pytorch_home_path():
    return os.path.join(container_home_path(), "pytorch")

def container_tensorflow_home_path():
    return os.path.join(container_home_path(), "tensorflow")

def container_dll_home_path(dll):
    if dll == "torch":
        return container_pytorch_home_path()
    elif dll == "tf":
        return container_tensorflow_home_path()
    else:
        assert(False)

def container_cov_tool_cmd():
    cov_tool_path = os.path.join(container_home_path(), "coverage.py")
    return f"python3 -u {cov_tool_path}"

def union(dict1, dict2):
    d = {}
    for k, v in dict1.items():
        d[k] = v
    for k, v in dict2.items():
        assert(k not in d.keys())
        d[k] = v
    return d

def emphasize(s):
    line = "+" + ("-" * (len(s) + 2)) + "+\n"
    return line + "| " + s + " |\n" + line

def check(bexp, msg):
    if not bexp:
        print(emphasize(msg))
        exit(0)

container_env_vars_tf_gcov = [
    f"GCOV_PREFIX={container_tensorflow_home_path()}",
    f"GCOV_PREFIX_STRIP=3",
]
