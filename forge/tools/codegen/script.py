import generate_op as go
import generate_op_struct as gt
import generate_dtype as gd
import generate_device as gen_dev
import generate_type_def as gtf
import cpu
import xpu

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

output_path = ROOT_PRJ / "aten/dispatcher/_op.hpp"
with open(output_path, "w") as table:
    table.write(gt.code)

output_path = ROOT_PRJ / "aten/Aten/core/dtype.def"
with open(output_path, "w") as def_ostream:
    def_ostream.write(gtf.code)

cpu.write()
xpu.write()
