import generate_op as go
import generate_table as gt
import generate_cpu_kn as gck
import generate_xpu_kn as gxk
import generate_yaml as gy
import generate_map as gm

output_path = "D:/TensorForge/aten/dispatcher/register.cpp"
with open(output_path, "w") as register:
    register.write(go.code)

output_path = "D:/TensorForge/aten/dispatcher/_table.hpp"
with open(output_path, "w") as table:
    table.write(gt.code)

output_path ="D:/TensorForge/aten/Aten/native/cpu/core/cpu_kernels.hpp"
with open(output_path, "w") as cpu_kernels:
    cpu_kernels.write(gck.code)

output_path = "D:/TensorForge/aten/Aten/native/xpu/core/xpu_kernels.hpp"
with open(output_path, "w") as xpu_kernels:
    xpu_kernels.write(gxk.code)

gy.run()

output_path = "D:/TensorForge/aten/dispatcher/_map.hpp"
with open(output_path, "w") as dttb:
    dttb.write(gm.code)
