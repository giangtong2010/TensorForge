import yaml
import load_yaml as loader
from jinja2 import Environment, FileSystemLoader

config = loader.load_yaml("yaml/operators.yaml")

env = Environment(loader=FileSystemLoader("register"))
temp = env.get_template("generate_cpu_kn.hpp.jinja")
code = temp.render(**config)
