import yaml
from jinja2 import Environment, FileSystemLoader

with open("yaml/operators.yaml") as op:
    config = yaml.safe_load(op)

env = Environment(loader=FileSystemLoader("register"))
temp = env.get_template("generate_cpu_kn.hpp.jinja")
code = temp.render(
    operators=config["operators"],
    backends=config["backends"]
)
