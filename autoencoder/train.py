import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from utils import calculate_kurtosis
import torch.optim as optim
from model import Encoder, Decoder
import torch.nn as nn
import torch.nn.functional as F
import zarr

# Assuming 'dataset' is your dataset
dataset = zarr.load('data/data.zarr')
dataset = torch.tensor(dataset, dtype=torch.float32)  # Ensure dataset is a PyTorch tensor

# Define the dataset sizes
num_samples = len(dataset)
train_size = int(0.7 * num_samples)
val_size = int(0.1 * num_samples)
test_size = num_samples - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoader instances
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate encoders and decoder
encod_f = Encoder()
encod_m = Encoder()
decoder = Decoder()

# Define optimizer
optimizer = optim.Adam(list(encod_f.parameters()) + list(encod_m.parameters()) +
                           list(decoder.parameters()), lr=0.01, weight_decay=0.001)

num_epochs = 10
best_val_loss = float('inf')  # Initialize best validation loss to infinity

# Training loop
for epoch in range(num_epochs):
    encod_f.train()
    encod_m.train()
    decoder.train()
    
    train_loss = 0.0
    for patches in train_dataloader:
        optimizer.zero_grad()
        
        # Forward pass through encoders
        encoded_fetal = encod_f(patches)
        encoded_maternal = encod_m(patches)
        
        # Calculate kurtosis
        k_f = calculate_kurtosis(encoded_fetal)
        k_m = calculate_kurtosis(encoded_maternal)
        kurtosis_loss = torch.mean(torch.square((torch.abs(k_m) - torch.abs(k_f))))
        
        # Forward pass through decoder
        reconstructed = decoder(encoded_fetal + encoded_maternal)
        
        # Calculate loss
        loss = F.mse_loss(reconstructed, patches) - 0.1 * kurtosis_loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        print(loss.item())
        train_loss += loss.item()
    
    # Validation step
    encod_f.eval()
    encod_m.eval()
    decoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for patches in val_dataloader:
            encoded_fetal = encod_f(patches)
            encoded_maternal = encod_m(patches)
            k_f = calculate_kurtosis(encoded_fetal)
            k_m = calculate_kurtosis(encoded_maternal)
            kurtosis_loss = torch.mean(torch.square((torch.abs(k_m) - torch.abs(k_f))))
            reconstructed = decoder(encoded_fetal + encoded_maternal)
            loss = F.mse_loss(reconstructed, patches) - 0.1 * kurtosis_loss
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_dataloader):.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Check if the current validation loss is the best we've seen so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Save the model state
        torch.save({
            'encod_f_state_dict': encod_f.state_dict(),
            'encod_m_state_dict': encod_m.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'best_model.pth')
        print(f'Best model saved with validation loss: {best_val_loss:.4f}')

# Testing step
encod_f.eval()
encod_m.eval()
decoder.eval()
test_loss = 0.0
with torch.no_grad():
    for patches in test_dataloader:
        encoded_fetal = encod_f(patches)
        encoded_maternal = encod_m(patches)
        k_f = calculate_kurtosis(encoded_fetal)
        k_m = calculate_kurtosis(encoded_maternal)
        kurtosis_loss = torch.mean(torch.square((torch.abs(k_m) - torch.abs(k_f))))
        reconstructed = decoder(encoded_fetal + encoded_maternal)
        loss = F.mse_loss(reconstructed, patches) - 0.1 * kurtosis_loss
        test_loss += loss.item()

print(f'Test Loss: {test_loss/len(test_dataloader):.4f}')
