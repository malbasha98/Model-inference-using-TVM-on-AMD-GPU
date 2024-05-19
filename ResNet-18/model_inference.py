from tvm.contrib import graph_executor
from model_compile import *


lib_path = get_lib_path()

target = "rocm --host=llvm --libs=miopen"
dev = tvm.device(str(target), 0)

num_of_images = int(input("Enter a number of images for inference:"))
print("Number of images for inference: ", num_of_images)

np.random.seed(0)


def load_compiled_model():
    loaded_lib = tvm.runtime.load_module(lib_path)
    m = graph_executor.GraphModule(loaded_lib["default"](dev))
    return m


module = load_compiled_model()

images = load_data()
image = np.expand_dims(images[0], 0)
print(images.shape)

input_name = "x"
shape_dict = dict()
for i in range(0, num_of_images):
    shape_dict.update({'x' + str(i): image.shape})

input_names = [str(input_name) for input_name in shape_dict.keys()]
input_dict = dict(zip(input_names, np.expand_dims(images, 1)))

dtype = "float32"
module.set_input(**input_dict)

# evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, number=100, repeat=3))

module.run()

output_shape = (num_of_images, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
print(module.get_output(0, tvm.nd.empty(output_shape)).numpy().shape)
