import yaml
import read_cpp

def run():
    output_path = "D:/TensorForge/forge/tools/codegen/yaml/utils.yaml"

    with open(output_path, "w") as file:
        yaml.dump(
            read_cpp.backend, 
            file,
            sort_keys=False,
            default_flow_style=False
        )
        yaml.dump(
            read_cpp.dtype,
            file,
            sort_keys=False,
            default_flow_style=False
        )
