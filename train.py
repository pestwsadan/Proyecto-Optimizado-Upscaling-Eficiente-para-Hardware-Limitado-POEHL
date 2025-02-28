import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os

# Modelo con factor de upscale=2
class MiniESPCN(nn.Module):
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3 * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

# Dataset con transformaciones separadas
class SRDataset(torch.utils.data.Dataset):
    def __init__(self, lr_dir, hr_dir, lr_size=(180, 320), hr_size=(360, 640)):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.filenames = os.listdir(lr_dir)
        
        self.lr_transform = transforms.Compose([
            transforms.Resize(lr_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        lr_img = Image.open(os.path.join(self.lr_dir, self.filenames[idx]))
        hr_img = Image.open(os.path.join(self.hr_dir, self.filenames[idx]))
        
        lr_img = self.lr_transform(lr_img)
        hr_img = self.hr_transform(hr_img)
        
        return lr_img, hr_img

# Configuración
train_lr_dir = r"F:\Nueva carpeta (2)\ia\data\train\lr"
train_hr_dir = r"F:\Nueva carpeta (2)\ia\data\train\hr"
train_dataset = SRDataset(train_lr_dir, train_hr_dir, lr_size=(180, 320), hr_size=(360, 640))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniESPCN(upscale_factor=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()

# Entrenamiento
num_epochs = 50
for epoch in range(num_epochs):
    for lr, hr in train_loader:
        lr = lr.to(device)
        hr = hr.to(device)
        
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "miniespcn.pth") 