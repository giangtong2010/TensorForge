import yaml
from jinja2 import Environment, FileSystemLoader

with open("yaml/operators.yaml") as f:
    config = yaml.safe_load(f)

env = Environment(loader=FileSystemLoader("register"))
template = env.get_template("register.hpp.jinja")

code = template.render(
    operators=config["operators"]
)
