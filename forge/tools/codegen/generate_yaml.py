import yaml
import read_cpp

def run():
    output_path = "D:/TensorForge/forge/tools/codegen/yaml/backends.yaml"

    with open(output_path, "w") as file:
        yaml.dump(
            read_cpp.backend, 
            file,
            sort_keys=False,
            default_flow_style=False
        )
    
    output_path = "D:/TensorForge/forge/tools/codegen/yaml/dtypes.yaml"
    with open(output_path, "w") as file:
        yaml.dump(
            read_cpp.dtype,
            file,
            sort_keys=False,
            default_flow_style=False
        )
