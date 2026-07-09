import yaml
import load_yaml as loader
from jinja2 import Environment, FileSystemLoader

config=loader.load_yaml("yaml/operators.yaml")

env = Environment(loader=FileSystemLoader("register"))
template = env.get_template("register.hpp.jinja")

code = template.render(**config)
