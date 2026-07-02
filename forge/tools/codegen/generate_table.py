import yaml
from jinja2 import FileSystemLoader, Environment

with open("yaml/operators.yaml") as op:
    config = yaml.safe_load(op)

env = Environment(loader=FileSystemLoader("register"))
reg = env.get_template("table.hpp.jinja")
code = reg.render(
    operators=config["operators"],
    backends=config["backends"]
)
