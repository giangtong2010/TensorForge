import yaml
from jinja2 import FileSystemLoader, Environment

with open("yaml/operators.yaml") as op:
    config = yaml.safe_load(op)

ops = []
env = Environment(loader=FileSystemLoader("register"))
reg = env.get_template("op.hpp.jinja")

for category_name, category in config.items():
    if category_name == "include":
        continue

    if isinstance(category, list):
        ops.extend(op["name"] for op in category)
    else:
        for group in category.values():
            ops.extend(op["name"] for op in group)

code = reg.render(
    operators=ops
)
