import scipy.stats as stats
import torch
import numpy as np

def calculate_kurtosis(tensor):
    tensor_np = tensor.cpu().detach().numpy()
    kurtosis_values = stats.kurtosis(tensor_np, axis=1, nan_policy='omit')
    # Replace NaNs with 0
    kurtosis_values = np.nan_to_num(kurtosis_values, nan=0.0)
    return torch.tensor(kurtosis_values, dtype=torch.float32, requires_grad=True)

def calculate_psnr(original, reconstructed, max_val=1.0):
    mse = F.mse_loss(original, reconstructed)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr
