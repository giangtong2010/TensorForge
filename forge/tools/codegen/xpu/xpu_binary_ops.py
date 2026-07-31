import load_yaml as loader
from jinja2 import Environment

def xpu_binary_ops(yaml_path, env: Environment):
    config = loader.load_yaml(yaml_path)
    temp = env.get_template("generate_xpu_binary.hpp.jinja")
    code = temp.render(
        operators=config["element_wise"]["binary_ops"],
        backends=config["backends"],
        dtypes=config["dtypes"]
    )
    return code
