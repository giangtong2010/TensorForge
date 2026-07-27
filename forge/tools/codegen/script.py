import generate_op as go
import generate_table as gt
import generate_cpu_kn as gck
import generate_xpu_kn as gxk
import generate_dtype as gd
import generate_device as gen_dev

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOT_PRJ = ROOT.parent.parent.parent

output_path = ROOT_PRJ / "cpp20/core/_Dtype.hpp"
with open(output_path, "w") as dtype:
    dtype.write(gd.code)

output_path = ROOT_PRJ / "cpp20/core/_Device.hpp"
with open(output_path, "w") as device:
    device.write(gen_dev.code)

output_path = ROOT_PRJ / "aten/dispatcher/register.cpp"
with open(output_path, "w") as register:
    register.write(go.code)

output_path = ROOT_PRJ / "aten/dispatcher/_table.hpp"
with open(output_path, "w") as table:
    table.write(gt.code)

output_path = ROOT_PRJ / "aten/Aten/native/cpu/core/cpu_kernels.hpp"
with open(output_path, "w") as cpu_kernels:
    cpu_kernels.write(gck.code)

output_path = ROOT_PRJ / "aten/Aten/native/xpu/core/xpu_kernels.hpp"
with open(output_path, "w") as xpu_kernels:
    xpu_kernels.write(gxk.code)
