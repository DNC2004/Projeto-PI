import subprocess
import sys
# Verificar se o PyTorch já está instalado -- TEMPORÁRIO??
try:
    import torch 
    import torchvision

except ImportError:
    print("\nDEBUG -- Bibliotecas não encontradas...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\nDEBUG -- Imports instalados!")
    
    except Exception as e:
        print(f"\nDEBUG -- Falha ao instalar: {e}")
#### IMPORTS
### Processamento
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import hashlib
###

### CNNs
import torch
import torchvision

import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
from uuid import uuid4
###

print("DEBUG -- Bibliotecas importadas com sucesso")



# Load main image / FUNCIONA!
image_path = 'imagem_final.png'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"A imagem '{image_path}' não existe...")

# Debug
output_folder = "imagens_teste"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

img_teste = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
file_path = os.path.join(output_folder, "imagem_teste.png")
cv2.imwrite(file_path, img_teste)
# END

img_h, img_w = img.shape[:2]

# Extract digits via HSV mask (gold color)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h = img_hsv[:, :, 0]
s = img_hsv[:, :, 1]
v = img_hsv[:, :, 2]

# Debug
print(f"DEBUG -- Valores de HSV da imagem:\nH:{h.mean()}\nS:{s.mean()}\nV:{v.mean()}")
file_path = os.path.join(output_folder, "imagem_teste_hsv.png")
cv2.imwrite(file_path, img_hsv)
# END

if v.mean() > 190: # Solução temporária para a imagem com os brancos mais claros funcionar
    lower_gold = np.array([20, 60, 75], np.uint8)
    upper_gold = np.array([65, 255, 255], np.uint8)
    gold_mask = cv2.inRange(img_hsv, lower_gold, upper_gold)

else:
    #Mascaras Para deter os nums / FUNCIONA!
    lower_gold = np.array([10, 60, 30], np.uint8)
    upper_gold = np.array([150,255, 255], np.uint8)
    gold_mask = cv2.inRange(img_hsv, lower_gold, upper_gold)

kernel = np.ones((4, 4), np.uint8)

gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
gold_mask = cv2.dilate(gold_mask, kernel, iterations=1)

# Imagem 
img_binary = gold_mask.copy()
# Debug
file_path = os.path.join(output_folder, "imagem_teste_binaria.png")
cv2.imwrite(file_path, img_binary)
# END

# 
templates = {}
for i in range(1, 16):
    path = f"../templates/mask{i}.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    templates[i] = img
    
def augment(img):
    h, w = img.shape

    # Random scaling
    scale = random.uniform(0.8, 1.2)
    img = cv2.resize(img, None, fx=scale, fy=scale)

    # Random translation
    dx = random.randint(-5, 5)
    dy = random.randint(-5, 5)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=0)

    # Random rotation
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=0)

    # Morphological noise
    if random.random() < 0.5:
        k = random.randint(1, 3)
        img = cv2.erode(img, np.ones((k,k), np.uint8))
    else:
        k = random.randint(1, 3)
        img = cv2.dilate(img, np.ones((k,k), np.uint8))

    # Resize to final size
    img = cv2.resize(img, (60, 60))

    return img

os.makedirs("dataset", exist_ok=True)

for label, template in templates.items():
    class_dir = f"dataset/{label}"
    os.makedirs(class_dir, exist_ok=True)

    for i in range(1000):
        img = augment(template)
        cv2.imwrite(f"{class_dir}/{uuid4().hex}.png", img)
        
empty_dir = "dataset/0"
os.makedirs(empty_dir, exist_ok=True)

for i in range(1500):
    noise = np.random.randint(0, 30, (60, 60), dtype=np.uint8)
    cv2.imwrite(f"{empty_dir}/{uuid4().hex}.png", noise)