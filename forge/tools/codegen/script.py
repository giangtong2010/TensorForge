import generate_op as go
import generate_table as gt

output_path = "register/register.cpp"
with open(output_path, "w") as register:
    register.write(go.code)

output_path = "D:/TensorForge/aten/dispatcher/_table.hpp"
with open(output_path, "w") as table:
    table.write(gt.code)
