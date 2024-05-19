from tvm.contrib.download import download_testdata
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os

LIB_PATH = os.path.join(os.getcwd(), "model.tar")

token_predictor = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True).eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

if __name__ == "__main__":
    
    random_tokens = torch.randint(10000, (5,))

    traced_token_predictor = torch.jit.trace(token_predictor, random_tokens)

    inputs = [("dummy_input_name", (5,))]

    mod, params = relay.frontend.from_pytorch(traced_token_predictor, inputs, default_dtype="int64")
    print(mod)

    target = "rocm --host=llvm --libs=miopen"

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    lib.export_library(LIB_PATH) 

