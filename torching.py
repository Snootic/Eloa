import torch

print("torch threads:", torch.get_num_threads())
print("interop threads:", torch.get_num_interop_threads())

print("setting torch threads to 4")
torch.set_num_threads(4)
