import load_yaml as loader
from jinja2 import Environment

def xpu_memory_kn(yaml_path, env: Environment):
    config = loader.load_yaml(yaml_path)
    temp = env.get_template("generate_xpu_memory_kn.hpp.jinja")
    code = temp.render(
        operators=config["memory"]["memory_kn"],
        dtypes=config["dtypes"],
        backends=config["backends"]
    )
    return code
