from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import sys

from .cpu_binary_ops import *
from .cpu_memory_kn import *
from .cpu_memory_ops import *
from .cpu_linalg_ops import *

modules = [
    f"{name.split(".")[-1]}.hpp"
    for name in sys.modules
    if name.startswith("cpu.") and "script" not in name
]

FILE_PATH = Path(__file__).resolve().parent
YAML_PATH = FILE_PATH.parent / "yaml" / "operators.yaml"
_ROOT_PRJ = FILE_PATH.parent.parent.parent.parent

_env = Environment(loader=FileSystemLoader(str(FILE_PATH / "cpu_register")))
_path = "aten/Aten/native/cpu/core"

def write():
    with open(_ROOT_PRJ / _path / modules[0], "w") as file:
        file.write(cpu_binary_ops(YAML_PATH, _env))

    with open(_ROOT_PRJ / _path / modules[1], "w") as file:
        file.write(cpu_memory_kn(YAML_PATH, _env))

    with open(_ROOT_PRJ / _path / modules[2], "w") as file:
        file.write(cpu_memory_ops(YAML_PATH, _env))

    with open(_ROOT_PRJ / _path / modules[3], "w") as file:
        file.write(cpu_linalg_ops(YAML_PATH, _env))
