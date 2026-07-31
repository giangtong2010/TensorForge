import load_yaml as loader
import yaml
from jinja2 import Environment, FileSystemLoader

config = loader.load_yaml("yaml/operators.yaml")
with open("yaml/operators.yaml") as file:
    _config = yaml.safe_load(file)

ops = []
env = Environment(loader=FileSystemLoader("register"))
reg = env.get_template("register.hpp.jinja")

for category_name, category in _config.items():
    if category_name == "include":
        continue

    if isinstance(category, list):
        ops.extend(op["name"] for op in category)
    else:
        for group in category.values():
            ops.extend(op["name"] for op in group)

code = reg.render(
    operators=ops,
    dtypes=config["dtypes"],
    backends=config["backends"]
)
