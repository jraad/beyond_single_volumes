import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributions as dist
from torchsummary import summary
import os
import re
import numpy as np


class Data3D(Dataset):
    def __init__(self, file_dir, indices, t2_only=False, t1_only=False):
        self.file_dir = file_dir
        t1_list = sorted([x for x in os.listdir(self.file_dir) if x.endswith("T1w.npy")])
        t2_list = sorted([x for x in os.listdir(self.file_dir) if x.endswith("T2w.npy")])
        self.file_list = [y for x, y in enumerate(zip(t1_list, t2_list)) if x in indices]
        self.t2_only = t2_only
        self.t1_only = t1_only
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.t2_only:
            t2_path = os.path.join(self.file_dir, self.file_list[idx][1])
            t2 = np.load(t2_path)
            return np.reshape(t2, (1, 256, 256, 256)).astype("float32")
        elif self.t1_only:
            t1_path = os.path.join(self.file_dir, self.file_list[idx][0])
            t1 = np.load(t1_path)
            return np.reshape(t1, (1, 256, 256, 256)).astype("float32")
        else:
            t1_path = os.path.join(self.file_dir, self.file_list[idx][0])
            t2_path = os.path.join(self.file_dir, self.file_list[idx][1])
            t1 = np.load(t1_path)
            t2 = np.load(t2_path)
            return np.stack([t1, t2]).astype("float32")
    
    def get_subject(self, name):
        t1_name = [x for x in self.file_list if name in x[0]][0]
        print(self.file_list.index(t1_name) + 1)

        
class Data3DSegT2(Dataset):
    def __init__(self, file_dir, seg_dir, indices):
        self.file_dir = file_dir
        self.seg_dir = seg_dir
        self.t1_list = sorted([x for x in os.listdir(self.file_dir) if x.endswith("T1w.npy")])
        self.t2_list = sorted([x for x in os.listdir(self.file_dir) if x.endswith("T2w.npy")])
        self.file_list = [y for x, y in enumerate(zip(self.t1_list, self.t2_list)) if x in indices]
        self.seg_list =  [x.split("_T2w")[0] + "_seg_T2w.npy" for x in [y[1] for y in self.file_list]]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        t2_path = os.path.join(self.file_dir, self.file_list[idx][1])
        t2 = np.load(t2_path)
        
        seg_path = os.path.join(self.seg_dir, self.seg_list[idx])
        seg = np.load(seg_path)
        
        return np.reshape(t2, (1, 256, 256, 256)), np.reshape(seg, (9, 256, 256, 256))
        

class SegMaskData(Dataset):
    def __init__(self, file_dir, indices):
        self.file_dir = file_dir
        self.file_list = [y for x, y in enumerate(sorted(os.listdir(self.file_dir))) if x in indices]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        seg_path = os.path.join(self.file_dir, self.file_list[idx])
        seg = np.load(seg_path)
        
        return seg

    
class ResnetEncoder(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        self.num_channels = num_channels
        self.kernel_size = 3
        self.strides = 2
        
        self.pass1 = nn.Sequential(
            # nn.BatchNorm3d(num_channels),
            nn.InstanceNorm3d(num_channels),
            nn.ReLU(),
            nn.Conv3d(
                num_channels,
                32,
                kernel_size = self.kernel_size,
                stride = self.strides,
                padding = 1
            ),
        )

        self.pass2 = nn.Sequential(
            # nn.BatchNorm3d(32),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(
                32,
                32,
                kernel_size = self.kernel_size,
                stride = self.strides ,
                padding = 1
            )
        )
        
        self.conv_bypass = nn.Conv3d(
            self.num_channels,
            32,
            kernel_size = self.kernel_size,
            stride = self.strides * 2
        )
        self.activation_bypass = nn.ReLU()
        
        
    def forward(self, x_in):
        x = self.pass1(x_in)
        x = self.pass2(x)
        
        x_bypass = self.conv_bypass(x_in)
        x = x + x_bypass
        x = self.activation_bypass(x)
        return x

class ResnetEncoderHalf(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        self.num_channels = num_channels
        self.kernel_size = 3
        self.strides = 2
        
        self.pass1 = nn.Sequential(
            # nn.BatchNorm3d(num_channels),
            nn.InstanceNorm3d(num_channels),
            nn.ReLU(),
            nn.Conv3d(
                num_channels,
                32,
                kernel_size = self.kernel_size,
                stride = self.strides,
                padding = 1
            ),
        )
        
        self.conv_bypass = nn.Conv3d(
            self.num_channels,
            32,
            kernel_size = self.kernel_size,
            stride = self.strides,
            padding = 1
        )
        self.activation_bypass = nn.ReLU()
        
        
    def forward(self, x_in):
        x = self.pass1(x_in)
        
        x_bypass = self.conv_bypass(x_in)
        x = x + x_bypass
        x = self.activation_bypass(x)
        return x

class UpConv3d(nn.Module):
    def __init__(self, ch_in, ch_out, k_size=1, stride=2, scale=4, align_corners=False):
        super(UpConv3d, self).__init__()
        self.up = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride = stride, padding=1),
            nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=align_corners),
            # nn.Upsample(scale_factor=scale, mode='nearest'),
        )
    def forward(self, x):
        return self.up(x)


class ResnetDecoder(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        self.num_channels = num_channels
        self.kernel_size = 3
        self.strides = 2
        
        self.pass1 = nn.Sequential(
            # nn.BatchNorm3d(num_channels),
            nn.InstanceNorm3d(num_channels),
            nn.ReLU(),
            UpConv3d(
                num_channels,
                32,
                k_size = self.kernel_size,
                stride = self.strides
            )
        )
        
        self.pass2 = nn.Sequential(
            # nn.BatchNorm3d(32),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            UpConv3d(
                32,
                32,
                k_size = self.kernel_size,
                stride = self.strides
            )
        )
        
        self.conv_bypass = nn.ConvTranspose3d(
            self.num_channels,
            32,
            kernel_size = 4,
            stride = self.strides * 2
        )
        self.relu_bypass = nn.ReLU()
        
    def forward(self, x_in):
        x = self.pass1(x_in)
        x = self.pass2(x)
        
        x_bypass = self.conv_bypass(x_in)
        x = x + x_bypass
        x = self.relu_bypass(x)
        
        return x


class ResnetDecoderHalf(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        self.num_channels = num_channels
        self.kernel_size = 3
        self.strides = 2
        
        self.pass1 = nn.Sequential(
            # nn.BatchNorm3d(num_channels),
            nn.InstanceNorm3d(num_channels),
            nn.ReLU(),
            UpConv3d(
                num_channels,
                32,
                k_size = self.kernel_size,
                stride = self.strides
            )
        )
        
        self.conv_bypass = nn.ConvTranspose3d(
            self.num_channels,
            32,
            kernel_size = 2,
            stride = self.strides
        )
        self.relu_bypass = nn.ReLU()
        
    def forward(self, x_in):
        x = self.pass1(x_in)
        
        x_bypass = self.conv_bypass(x_in)
        x = x + x_bypass
        x = self.relu_bypass(x)
        
        return x


class VariationalLatent(nn.Module):
    def __init__(self):
        super().__init__()
        
         
        self.flatten = nn.Flatten()
        self.norm1 = nn.InstanceNorm1d(16384)
        self.norm2 = nn.InstanceNorm1d(16384)
        self.mu = nn.Linear(16384, 512)
        self.sigma = nn.Linear(16384, 512)
        
        
        
        self.dense_out = nn.Linear(512, 16384)
        self.unflatten = nn.Unflatten(-1, (32, 8, 8, 8))
    
    def forward(self, x):
        x = self.flatten(x)
        
        mu = self.norm1(x)
        mu = self.mu(mu)
        
        sigma = self.norm2(x)
        sigma = self.sigma(sigma)
        
        std = torch.exp(sigma / 2)
        z = mu + std * torch.randn_like(std)
        
        x = self.dense_out((mu + sigma) / 2)
        x = self.unflatten(x)
        return x
    

class VAE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.model = nn.Sequential(
            ResnetEncoder(channels),
            ResnetEncoder(32),
            ResnetEncoderHalf(32),
            VariationalLatent(),
            ResnetDecoderHalf(32),
            ResnetDecoder(32),
            ResnetDecoder(32),
            nn.ConvTranspose3d(32, channels, 1)
        )
        
    def forward(self, x):
        x = self.model(x)
        
        return x


class VAESegment(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.model = nn.Sequential(
            ResnetEncoder(in_channels),
            ResnetEncoder(32),
            ResnetEncoderHalf(32),
            VariationalLatent(),
            ResnetDecoderHalf(32),
            ResnetDecoder(32),
            ResnetDecoder(32),
            nn.ConvTranspose3d(32, out_channels, 1)
        )
        
    def forward(self, x):
        x = self.model(x)
        
        return x


class VariationalLatentDistribution(nn.Module):
    def __init__(self):
        super().__init__()
        
         
        self.flatten = nn.Flatten()
        self.norm1 = nn.InstanceNorm1d(16384)
        self.norm2 = nn.InstanceNorm1d(16384)
        self.mu = nn.Linear(16384, 512)
        self.sigma = nn.Linear(16384, 512)
    
    def forward(self, x):
        x = self.flatten(x)
        
        mu = self.norm1(x)
        mu = self.mu(mu)
        
        sigma = self.norm2(x)
        sigma = self.sigma(sigma)
        
        # std = torch.exp(sigma / 2)
        # z = mu + std * torch.randn_like(std)

        return mu, sigma
    

class VAELatent(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.model = nn.Sequential(
            ResnetEncoder(channels),
            ResnetEncoder(32),
            ResnetEncoderHalf(32),
            VariationalLatentDistribution()
        )
        
    def forward(self, x):
        x = self.model(x)
        
        return x

# class ResnetEncoder(nn.Module):
#     def __init__(self, num_channels):
#         super().__init__()
        
#         self.num_channels = num_channels
#         self.kernel_size = 2
#         self.strides = 2
        
#         self.pass1 = nn.Sequential(
#             # nn.BatchNorm3d(num_channels),
#             nn.InstanceNorm3d(num_channels),
#             nn.ReLU(),
#             nn.Conv3d(
#                 num_channels,
#                 32,
#                 kernel_size = self.kernel_size,
#                 stride = self.strides
#             ),
#         )

#         self.pass2 = nn.Sequential(
#             # nn.BatchNorm3d(32),
#             nn.InstanceNorm3d(32),
#             nn.ReLU(),
#             nn.Conv3d(
#                 32,
#                 32,
#                 kernel_size = self.kernel_size,
#                 stride = self.strides 
#             )
#         )
        
#         self.conv_bypass = nn.Conv3d(
#             self.num_channels,
#             32,
#             kernel_size = self.kernel_size,
#             stride = self.strides * 2
#         )
#         self.activation_bypass = nn.ReLU()
        
        
#     def forward(self, x_in):
#         x = self.pass1(x_in)
#         x = self.pass2(x)
        
#         x_bypass = self.conv_bypass(x_in)
#         x = x + x_bypass
#         x = self.activation_bypass(x)
#         return x

# class ResnetEncoderHalf(nn.Module):
#     def __init__(self, num_channels):
#         super().__init__()
        
#         self.num_channels = num_channels
#         self.kernel_size = 2
#         self.strides = 2
        
#         self.pass1 = nn.Sequential(
#             # nn.BatchNorm3d(num_channels),
#             nn.InstanceNorm3d(num_channels),
#             nn.ReLU(),
#             nn.Conv3d(
#                 num_channels,
#                 32,
#                 kernel_size = self.kernel_size,
#                 stride = self.strides
#             ),
#         )
        
#         self.conv_bypass = nn.Conv3d(
#             self.num_channels,
#             32,
#             kernel_size = self.kernel_size,
#             stride = self.strides
#         )
#         self.activation_bypass = nn.ReLU()
        
        
#     def forward(self, x_in):
#         x = self.pass1(x_in)
        
#         x_bypass = self.conv_bypass(x_in)
#         x = x + x_bypass
#         x = self.activation_bypass(x)
#         return x


# class ResnetDecoder(nn.Module):
#     def __init__(self, num_channels):
#         super().__init__()
        
#         self.num_channels = num_channels
#         self.kernel_size = 2
#         self.strides = 2
        
#         self.pass1 = nn.Sequential(
#             # nn.BatchNorm3d(num_channels),
#             nn.InstanceNorm3d(num_channels),
#             nn.ReLU(),
#             nn.ConvTranspose3d(
#                 num_channels,
#                 32,
#                 kernel_size = self.kernel_size,
#                 stride = self.strides
#             ),
#         )
        
#         self.pass2 = nn.Sequential(
#             # nn.BatchNorm3d(32),
#             nn.InstanceNorm3d(32),
#             nn.ReLU(),
#             nn.ConvTranspose3d(
#                 32,
#                 32,
#                 kernel_size = self.kernel_size,
#                 stride = self.strides
#             )
#         )
        
#         self.conv_bypass = nn.ConvTranspose3d(
#             self.num_channels,
#             32,
#             kernel_size = 4,
#             stride = self.strides * 2
#         )
#         self.relu_bypass = nn.ReLU()
        
#     def forward(self, x_in):
#         x = self.pass1(x_in)
#         x = self.pass2(x)
        
#         x_bypass = self.conv_bypass(x_in)
#         x = x + x_bypass
#         x = self.relu_bypass(x)
        
#         return x


# class ResnetDecoderHalf(nn.Module):
#     def __init__(self, num_channels):
#         super().__init__()
        
#         self.num_channels = num_channels
#         self.kernel_size = 2
#         self.strides = 2
        
#         self.pass1 = nn.Sequential(
#             # nn.BatchNorm3d(num_channels),
#             nn.InstanceNorm3d(num_channels),
#             nn.ReLU(),
#             nn.ConvTranspose3d(
#                 num_channels,
#                 32,
#                 kernel_size = self.kernel_size,
#                 stride = self.strides
#             ),
#         )
        
#         self.conv_bypass = nn.ConvTranspose3d(
#             self.num_channels,
#             32,
#             kernel_size = 2,
#             stride = self.strides
#         )
#         self.relu_bypass = nn.ReLU()
        
#     def forward(self, x_in):
#         x = self.pass1(x_in)
        
#         x_bypass = self.conv_bypass(x_in)
#         x = x + x_bypass
#         x = self.relu_bypass(x)
        
#         return x


# class VariationalLatent(nn.Module):
#     def __init__(self):
#         super().__init__()
        
         
#         self.flatten = nn.Flatten()
#         self.norm1 = nn.InstanceNorm1d(16384)
#         self.norm2 = nn.InstanceNorm1d(16384)
#         self.mu = nn.Linear(16384, 512)
#         self.sigma = nn.Linear(16384, 512)
        
        
        
#         self.dense_out = nn.Linear(512, 16384)
#         self.unflatten = nn.Unflatten(-1, (32, 8, 8, 8))
    
#     def forward(self, x):
#         x = self.flatten(x)
        
#         mu = self.norm1(x)
#         mu = self.mu(mu)
        
#         sigma = self.norm2(x)
#         sigma = self.sigma(sigma)
        
#         std = torch.exp(sigma / 2)
#         z = mu + std * torch.randn_like(std)
        
#         x = self.dense_out((mu + sigma) / 2)
#         x = self.unflatten(x)
#         return x