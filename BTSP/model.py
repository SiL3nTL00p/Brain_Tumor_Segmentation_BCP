# @title
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class U_Net(nn.Module):
    """3D U-Net for Brain Tumor Segmentation

    Architecture:
    - Encoder: 4 levels of downsampling with convolutional blocks
    - Bottleneck: Deepest level with convolutional block
    - Decoder: 4 levels of upsampling with skip connections
    - Output: Segmentation map with configurable number of classes
    """

    def __init__(self, in_channels=4, out_channels=4):
        """
        Args:
            in_channels: Number of input channels (default 4 for BraTS: T1, T1ce, T2, FLAIR)
            out_channels: Number of output classes (default 4: background, necrotic, edema, enhancing)
        """
        super().__init__()

        # Encoder (Downsampling) - Level 1
        self.enc1_conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.enc1_bn1 = nn.BatchNorm3d(64)
        self.enc1_conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.enc1_bn2 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder - Level 2
        self.enc2_conv1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.enc2_bn1 = nn.BatchNorm3d(128)
        self.enc2_conv2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.enc2_bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder - Level 3
        self.enc3_conv1 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.enc3_bn1 = nn.BatchNorm3d(256)
        self.enc3_conv2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.enc3_bn2 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder - Level 4
        self.enc4_conv1 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.enc4_bn1 = nn.BatchNorm3d(512)
        self.enc4_conv2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.enc4_bn2 = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.bottleneck_bn1 = nn.BatchNorm3d(1024)
        self.bottleneck_conv2 = nn.Conv3d(1024, 1024, kernel_size=3, padding=1)
        self.bottleneck_bn2 = nn.BatchNorm3d(1024)

        # Decoder (Upsampling) - Level 4
        self.up4 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.dec4_conv1 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.dec4_bn1 = nn.BatchNorm3d(512)
        self.dec4_conv2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.dec4_bn2 = nn.BatchNorm3d(512)

        # Decoder - Level 3
        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec3_conv1 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.dec3_bn1 = nn.BatchNorm3d(256)
        self.dec3_conv2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.dec3_bn2 = nn.BatchNorm3d(256)

        # Decoder - Level 2
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2_conv1 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.dec2_bn1 = nn.BatchNorm3d(128)
        self.dec2_conv2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.dec2_bn2 = nn.BatchNorm3d(128)

        # Decoder - Level 1
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1_conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.dec1_bn1 = nn.BatchNorm3d(64)
        self.dec1_conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.dec1_bn2 = nn.BatchNorm3d(64)

        # Output layer
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def _conv_block(self, x, conv1, bn1, conv2, bn2):
        """Helper method to apply a convolutional block"""
        x = conv1(x)
        x = bn1(x)
        x = self.relu(x)
        x = conv2(x)
        x = bn2(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        # Encoder with skip connections
        # Level 1
        enc1 = self._conv_block(x, self.enc1_conv1, self.enc1_bn1, self.enc1_conv2, self.enc1_bn2)
        x = self.pool1(enc1)

        # Level 2
        enc2 = self._conv_block(x, self.enc2_conv1, self.enc2_bn1, self.enc2_conv2, self.enc2_bn2)
        x = self.pool2(enc2)

        # Level 3
        enc3 = self._conv_block(x, self.enc3_conv1, self.enc3_bn1, self.enc3_conv2, self.enc3_bn2)
        x = self.pool3(enc3)

        # Level 4
        enc4 = self._conv_block(x, self.enc4_conv1, self.enc4_bn1, self.enc4_conv2, self.enc4_bn2)
        x = self.pool4(enc4)

        # Bottleneck
        x = self._conv_block(x, self.bottleneck_conv1, self.bottleneck_bn1, self.bottleneck_conv2, self.bottleneck_bn2)

        # Decoder with skip connections
        # Level 4
        x = self.up4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self._conv_block(x, self.dec4_conv1, self.dec4_bn1, self.dec4_conv2, self.dec4_bn2)

        # Level 3
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self._conv_block(x, self.dec3_conv1, self.dec3_bn1, self.dec3_conv2, self.dec3_bn2)

        # Level 2
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self._conv_block(x, self.dec2_conv1, self.dec2_bn1, self.dec2_conv2, self.dec2_bn2)

        # Level 1
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self._conv_block(x, self.dec1_conv1, self.dec1_bn1, self.dec1_conv2, self.dec1_bn2)

        # Output
        x = self.final_conv(x)

        return x