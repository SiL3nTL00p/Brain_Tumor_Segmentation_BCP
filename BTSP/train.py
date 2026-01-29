import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import U_Net
from loss_function import DiceLoss
from torch.utils.data import DataLoader
from data_loader import BraTSMRIDataset

# checks wether the gpu is avaialble or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loads the dataset for the training
dataset = BraTSMRIDataset("/content/brats2021", augment=True)
loader = DataLoader(dataset, batch_size=1, shuffle=True) # Further reduced batch size to 1

# Clear previous model and cache if they exist
if 'model' in locals() and model is not None:
    del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# training
model = U_Net(out_channels=1).to(device) 
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
epochs = 10

# Initialize variables for plotting
epoch_losses = []
batch_losses = []

for epoch in range(epochs):
    epoch_loss = 0.0
    batch_count = 0
    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        model.train()

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_losses.append(loss.item())
        batch_count += 1

        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(loader)}], Loss: {loss.item():.4f}")


    # Calculate average loss for epoch
    avg_epoch_loss = epoch_loss / batch_count
    epoch_losses.append(avg_epoch_loss)
    print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_epoch_loss:.4f}\n")

print("="*60)
print("Training Complete!")
print("="*60 + "\n")

# Plotting
fig = plt.figure(figsize=(16, 10))

# Create a 2x2 grid for plots
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

# Plot 1: Loss per epoch (with trend line)
ax1.plot(epoch_losses, marker='o', linewidth=2.5, markersize=8, color='#2E86AB', label='Epoch Loss')
ax1.fill_between(range(len(epoch_losses)), epoch_losses, alpha=0.3, color='#2E86AB')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Dice Loss', fontsize=12, fontweight='bold')
ax1.set_title('Average Loss per Epoch', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(range(len(epoch_losses)))
ax1.legend()

# Plot 2: Loss statistics
stats_text = f"""
Training Statistics:
━━━━━━━━━━━━━━━━━━━━━
Total Epochs: {epochs}
Total Batches: {len(batch_losses)}
Final Loss: {epoch_losses[-1]:.6f}
Best Loss: {min(epoch_losses):.6f}
Worst Loss: {max(epoch_losses):.6f}
Avg Loss: {sum(epoch_losses)/len(epoch_losses):.6f}

Device: {device}
Batch Size: {loader.batch_size}
Learning Rate: 0.001
Optimizer: Adam
Loss Function: Dice Loss
"""
ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.axis('off')

# Plot 3: Loss per batch (with moving average)
window_size = 10
moving_avg = []
for i in range(len(batch_losses)):
    if i < window_size:
        moving_avg.append(sum(batch_losses[:i+1]) / (i+1))
    else:
        moving_avg.append(sum(batch_losses[i-window_size+1:i+1]) / window_size)

ax3.plot(batch_losses, linewidth=0.8, color='#A23B72', alpha=0.5, label='Batch Loss')
ax3.plot(moving_avg, linewidth=2.5, color='#F18F01', label=f'Moving Average (window={window_size})')
ax3.set_xlabel('Batch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Dice Loss', fontsize=12, fontweight='bold')
ax3.set_title('Loss per Batch (All Epochs)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='upper right')

plt.suptitle('Brain Tumor Segmentation - Training Progress', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('training_loss_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("Plot saved as 'training_loss_plot.png'")
print("="*60)