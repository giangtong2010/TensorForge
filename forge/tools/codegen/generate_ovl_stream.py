import load_yaml as loader
from jinja2 import FileSystemLoader, Environment

config = loader.load_yaml("yaml/dtypes.yaml")
config2 = loader.load_yaml("yaml/backends.yaml")
env = Environment(loader=FileSystemLoader("register"))
temp = env.get_template("generate_ovl_stream.jinja")
code = temp.render(
    dtypes=config["dtypes"],
    backends=config2["backends"]
)
