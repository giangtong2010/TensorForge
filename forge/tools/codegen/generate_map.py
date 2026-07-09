import load_yaml as loader
from jinja2 import FileSystemLoader, Environment

config = loader.load_yaml("yaml/backends.yaml")
config.update(loader.load_yaml("yaml/dtypes.yaml"))

env = Environment(loader=FileSystemLoader("register"))
temp = env.get_template("generate_map.hpp.jinja")
code = temp.render(**config)
