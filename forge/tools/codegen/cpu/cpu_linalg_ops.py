import load_yaml as loader
from jinja2 import Environment

def cpu_linalg_ops(yaml_path, env: Environment):
    config = loader.load_yaml(yaml_path)
    temp = env.get_template("generate_cpu_linalg.hpp.jinja")
    code = temp.render(
        operators=config["linalg_ops"],
        dtypes=config["dtypes"],
        backends=config["backends"]
    )
    return code
