import generate_op as go
import generate_table as gt
import generate_cpu_kn as gck
import generate_xpu_kn as gxk

output_path = "D:/TensorForge/aten/dispatcher/register.cpp"
with open(output_path, "w") as register:
    register.write(go.code)

output_path = "D:/TensorForge/aten/dispatcher/_table.hpp"
with open(output_path, "w") as table:
    table.write(gt.code)

output_path ="D:/TensorForge/aten/Aten/native/core/cpu/cpu_kernels.hpp"
with open(output_path, "w") as cpu_kernels:
    cpu_kernels.write(gck.code)

output_path = "D:/TensorForge/aten/Aten/native/core/xpu/xpu_kernels.hpp"
with open(output_path, "w") as xpu_kernels:
    xpu_kernels.write(gxk.code)
