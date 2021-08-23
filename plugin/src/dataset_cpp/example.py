import torch
import dataset_cpp.code as code

def bar(x):
    print("example function in python calling c++ code")
    return code.test_func(torch.Tensor([1,2,3]), x)