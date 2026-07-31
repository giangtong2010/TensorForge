import load_yaml as loader
from jinja2 import Environment

def cpu_memory_ops(yaml_path, env: Environment):
    config = loader.load_yaml(yaml_path)
    temp = env.get_template("generate_cpu_memory.hpp.jinja")
    code = temp.render(
        operators=config["memory"]["memory_ops"],
        dtypes=config["dtypes"],
        backends=config["backends"]
    )
    return code
