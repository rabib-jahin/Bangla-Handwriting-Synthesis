import torch
print(torch.version.cuda)      # Shows CUDA version PyTorch was built with
print(torch.backends.cudnn.version())  # cuDNN version (if available)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")
