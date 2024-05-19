import onnx
import tvm.relay as relay
import tvm
import logging
import sys
from data_loader import *

torch.cuda.empty_cache()
torch.cuda.synchronize()

MODEL_PATH = os.path.join(os.getcwd(), "resnet18_Opset18.onnx")
TARGET = "rocm --host=llvm"
LIB_PATH = os.path.join(os.getcwd(), "model.tar")
BATCH_SIZE = get_num_of_images()


def get_lib_path():
    return LIB_PATH


def compile_and_save_module(model_path=MODEL_PATH, target=TARGET, _logging=False, use_miopen=False,
                            batch_size=BATCH_SIZE):
    onnx_model = onnx.load(model_path)
    input_name = "x"
    shape_dict = {input_name: (batch_size, 3, 224, 224)}

    if _logging:
        logging.getLogger("te_compiler").setLevel(logging.INFO)
        logging.getLogger("te_compiler").addHandler(logging.StreamHandler(sys.stdout))

    if use_cudnn:
        target += " --libs=miopen"

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
        lib.export_library(LIB_PATH)


if __name__ == "__main__":
    compile_and_save_module(_logging=False, batch_size=BATCH_SIZE)
