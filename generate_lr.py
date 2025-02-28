import cv2
import os

# Configuración (¡Ajustado a upscale_factor=2!)
scale_factor = 2  # Reducción 2x (ej: 1080p → 540p)
input_dir = r"F:\Nueva carpeta (2)\ia\data\train\hr"
output_dir = r"F:\Nueva carpeta (2)\ia\data\train\lr"

# Crear carpeta LR si no existe
os.makedirs(output_dir, exist_ok=True)

# Verificar archivos en HR
if not os.listdir(input_dir):
    print(f"ERROR: La carpeta {input_dir} está vacía.")
    print("¡Descomprime DIV2K_train_HR.zip aquí!")
    exit()

# Procesar imágenes
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        hr_path = os.path.join(input_dir, filename)
        hr_img = cv2.imread(hr_path)
        
        if hr_img is None:
            print(f"Error al leer: {hr_path}")
            continue
        
        # Generar LR (2x más pequeña)
        h, w = hr_img.shape[:2]
        lr_img = cv2.resize(
            hr_img, 
            (w // scale_factor, h // scale_factor),  # 1920x1080 → 960x540
            interpolation=cv2.INTER_CUBIC
        )
        
        # Guardar
        lr_path = os.path.join(output_dir, filename)
        cv2.imwrite(lr_path, lr_img)
        print(f"Generado: {lr_path}")