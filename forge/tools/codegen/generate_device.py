import load_yaml as loader
from jinja2 import Environment, FileSystemLoader

config = loader.load_yaml("yaml/backends.yaml")

env = Environment(loader=FileSystemLoader("register"))
temp = env.get_template("generate_device.hpp.jinja")
code = temp.render(
    backends=config["backends"]
)
