import torch
from model import MiniESPCN
from torchvision import transforms
from PIL import Image

# Cargar el modelo entrenado
model = MiniESPCN(upscale_factor=2)
model.load_state_dict(torch.load("miniespcn.pth"))
model.eval()

# Transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Cargar una imagen de baja resolución
lr_image = Image.open("test_image_lr.png")
lr_tensor = transform(lr_image).unsqueeze(0)  # Añadir dimensión de lote

# Inferencia
with torch.no_grad():
    sr_tensor = model(lr_tensor)

# Guardar la imagen de super-resolución
sr_image = transforms.ToPILImage()(sr_tensor.squeeze(0).clamp(-1, 1) * 0.5 + 0.5)
sr_image.save("test_image_sr.png")