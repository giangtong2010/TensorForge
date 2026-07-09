import yaml
from pathlib import Path

visited = []

def load_yaml(path):
    result = {}
    path = Path(path)

    if path in visited:
        raise RuntimeError(f"include cycled in file {path}")
    visited.append(path)

    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file) or {}

        if "include" in config:
            lis_inc = config["include"]

            for file_inc in lis_inc:
                include_path = path.parent / file_inc
                ans = load_yaml(include_path)
                result.update(ans)

        config.pop("include", None)
        result.update(config)
        return result
    
    finally:
        visited.remove(path)
