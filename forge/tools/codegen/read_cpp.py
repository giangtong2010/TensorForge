import os
from clang import cindex
from clang.cindex import CursorKind

pathes = [
    "D:/TensorForge/cpp20/core/_Device.hpp",
    "D:/TensorForge/cpp20/core/_Dtype.hpp"
]
index = cindex.Index.create()

backend = {}
dtype = {}

for path in pathes:
    tu = index.parse(
        path=path,
        args=[
            "-std=c++20"
        ]
    )
    source = str(tu.spelling)

    for node in tu.cursor.walk_preorder():
        if (
            node.kind == CursorKind.ENUM_DECL 
            and node.location.file != None 
            and str(node.location.file) == source
        ):
            if str(node.spelling) == "DeviceType":
                lst_backend = []
                for child in node.get_children():
                    lst_backend.append(str(child.spelling).lower())

                backend.update({"backends": lst_backend})

            elif str(node.spelling) == "Dtype":
                lst_dtype = []
                for child in node.get_children():
                    lst_dtype.append(str(child.spelling).lower())
                
                dtype.update({"dtypes": lst_dtype})

print(backend)
print(dtype)
