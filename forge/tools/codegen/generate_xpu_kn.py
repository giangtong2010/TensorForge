import yaml
import load_yaml as loader
from jinja2 import Environment, FileSystemLoader

config = loader.load_yaml("yaml/operators.yaml")

env = Environment(loader=FileSystemLoader("register"))
tempalte = env.get_template("generate_xpu_kn.hpp.jinja")
code = tempalte.render(**config)
