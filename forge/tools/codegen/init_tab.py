import load_yaml as loader
from jinja2 import FileSystemLoader, Environment

config = loader.load_yaml("yaml/dtypes.yaml")

dtype_map = {d["name"]: d for d in config["dtypes"]}
dtype_map = {d["name"]: d for d in config["dtypes"]}

def find_dtype(category, bits):
    for dtype in config["dtypes"]:
        if (
            dtype["category"] == category
            and dtype.get("bits") == bits
        ):
            return dtype
    raise RuntimeError(f"Cannot find {category}({bits})")


def find_complex(base_name):
    for dtype in config["dtypes"]:
        if (
            dtype["category"] == "complex"
            and dtype["base"] == base_name
        ):
            return dtype
    raise RuntimeError(f"Cannot find complex type for {base_name}")


def next_signed(bits):
    if bits <= 8:
        return find_dtype("integral", 16)
    elif bits <= 16:
        return find_dtype("integral", 32)
    elif bits <= 32:
        return find_dtype("integral", 64)
    else:
        raise RuntimeError("No larger signed integer")


def promote(lhs, rhs):

    # same dtype
    if lhs["name"] == rhs["name"]:
        return lhs

    # bool
    if lhs["category"] == "bool":
        return rhs

    if rhs["category"] == "bool":
        return lhs

    # complex
    if lhs["category"] == "complex" or rhs["category"] == "complex":

        lhs_base = dtype_map[lhs["base"]] if lhs["category"] == "complex" else lhs
        rhs_base = dtype_map[rhs["base"]] if rhs["category"] == "complex" else rhs

        result = promote(lhs_base, rhs_base)

        return find_complex(result["name"])

    # floating + floating
    if lhs["category"] == rhs["category"] == "floating":
        return find_dtype(
            "floating",
            max(lhs["bits"], rhs["bits"])
        )

    # integral + floating
    cats = {lhs["category"], rhs["category"]}
    if cats == {"integral", "floating"}:
        return lhs if lhs["category"] == "floating" else rhs

    # integral + integral
    if lhs["category"] == rhs["category"] == "integral":

        # same signedness
        if lhs["signed"] == rhs["signed"]:
            return find_dtype(
                "integral",
                max(lhs["bits"], rhs["bits"])
            )

        # separate signed / unsigned
        signed = lhs if lhs["signed"] else rhs
        unsigned = rhs if lhs["signed"] else lhs

        if signed["bits"] > unsigned["bits"]:
            return signed

        if signed["bits"] == unsigned["bits"]:
            return next_signed(signed["bits"])

        return next_signed(unsigned["bits"])

    raise RuntimeError(
        f"Unsupported promotion: {lhs['name']} + {rhs['name']}"
    )

def init_tab():
    table = []
    for lhs in config["dtypes"]:
        row = []
        for rhs in config["dtypes"]:
            row.append(promote(lhs, rhs))
        table.append(row)
    return table
