from model_compile import *

target = "rocm --host=llvm --libs=miopen"
lib = tvm.runtime.load_module(LIB_PATH)
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

input_sentence = "Example input sentence that we are giving to GPT-2 model "
inputs = tokenizer(input_sentence, return_tensors="pt")
input_names = ["inputs"]
input_data = [inputs]

input_dict = {"inputs":inputs['input_ids']}

module.set_input(**input_dict)

print("Evaluate inference time cost...")
print(module.benchmark(dev, number=100, repeat=3))

print(module.get_output(0))
