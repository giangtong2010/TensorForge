import yaml
from jinja2 import FileSystemLoader, Environment

with open("yaml/operators.yaml") as op:
    config = yaml.safe_load(op)
with open("yaml/backends.yaml") as backends:
    _backends = yaml.safe_load(backends)
with open("yaml/dtypes.yaml") as dtype:
    _dtype = yaml.safe_load(dtype)

env = Environment(loader=FileSystemLoader("register"))
reg = env.get_template("table.hpp.jinja")
code = reg.render(
    operators=config["operators"],
    backends=_backends["backends"],
    dtypes=_dtype["dtypes"]
)
