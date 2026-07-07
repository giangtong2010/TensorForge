import yaml
from jinja2 import Environment, FileSystemLoader

with open("yaml/operators.yaml") as op:
    config = yaml.safe_load(op)

env = Environment(loader=FileSystemLoader("register"))
tempalte = env.get_template("generate_xpu_kn.hpp.jinja")
code = tempalte.render(
    operators=config["operators"],
    backends=config["backends"]
)
