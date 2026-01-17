import subprocess
import sys
import os
import cv2
import numpy as np
import random
from uuid import uuid4
from torchvision import transforms
from torchvision.datasets import MNIST


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

print("DEBUG -- Bibliotecas importadas com sucesso")

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import shutil
from sklearn.model_selection import train_test_split

# ============================================================
# Check and import PyTorch / torchvision
# ============================================================

# ============================================================
# IMAGE LOADING / MASKS (UNCHANGED)
# ============================================================
image_path = 'imagem_final.png'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"A imagem '{image_path}' não existe...")

output_folder = "imagens_teste"
os.makedirs(output_folder, exist_ok=True)

img_teste = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(output_folder, "imagem_teste.png"), img_teste)

img_h, img_w = img.shape[:2]

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]

print(f"DEBUG -- Valores de HSV da imagem:\nH:{h.mean()}\nS:{s.mean()}\nV:{v.mean()}")
cv2.imwrite(os.path.join(output_folder, "imagem_teste_hsv.png"), img_hsv)

if v.mean() > 190:
    lower_gold = np.array([20, 60, 75], np.uint8)
    upper_gold = np.array([65, 255, 255], np.uint8)
else:
    lower_gold = np.array([10, 60, 30], np.uint8)
    upper_gold = np.array([150,255, 255], np.uint8)

gold_mask = cv2.inRange(img_hsv, lower_gold, upper_gold)
kernel = np.ones((4, 4), np.uint8)
gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
gold_mask = cv2.dilate(gold_mask, kernel, iterations=1)
img_binary = gold_mask.copy()
cv2.imwrite(os.path.join(output_folder, "imagem_teste_binaria.png"), img_binary)

# ============================================================
# CNN PREPROCESSING
# ============================================================
mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

train_ds = MNIST(
    root="mnist_data",
    train=True,
    download=True,
    transform=mnist_transform
)

val_ds = MNIST(
    root="mnist_data",
    train=False,
    download=True,
    transform=mnist_transform
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)




def preprocess_for_cnn(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = 255 - img
    
    img = cv2.resize(img, (28, 28))
    img = img.astype(np.float32) / 255.0

    # MNIST-like normalization
    img = (img - 0.5) / 0.5

    return torch.from_numpy(img).unsqueeze(0)


# ============================================================
# CNN MODEL
# ============================================================
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10)  # 1 -- 9
        )
    def forward(self, x):
        return self.classifier(self.features(x))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ============================================================
# TRAIN ONCE / LOAD MODEL
# ============================================================
MODEL_PATH = "digit_cnn.pth"
if os.path.exists(MODEL_PATH):
    print("DEBUG -- Loading trained CNN...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    for x, y in train_loader:
        out = model(x)             # get predictions for this batch
        print("DEBUG -- Predicted classes:", out.argmax(1))  # should include 0–15
        break 
    SKIP_TRAINING = True
else:
    SKIP_TRAINING = False

if not SKIP_TRAINING:
    for epoch in range(30):  # fewer epochs for CPU
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Avg Loss {avg_loss:.4f}")
    torch.save(model.state_dict(), MODEL_PATH)

# ============================================================
# CNN PREDICTION
# ============================================================
def cnn_predict(roi):
    tensor = preprocess_for_cnn(roi).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    label = probs.argmax().item()
    confidence = probs[label].item()

    return label, confidence

# ============================================================
# GRID + CONTOUR + TWO-DIGIT HANDLING
# ============================================================
cell_h = img_binary.shape[0] // 4
cell_w = img_binary.shape[1] // 4
regioes_unidas = []

for row in range(4):
    for col in range(4):
        y1, y2 = row * cell_h, (row + 1) * cell_h
        x1, x2 = col * cell_w, (col + 1) * cell_w
        mask = img_binary[y1:y2, x1:x2]
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            regioes_unidas.append((x1, y1, cell_w, cell_h, None,[]))
            continue

        # Merge all points to get full bounding box
        pts = np.vstack([c.reshape(-1,2) for c in contornos])
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        margem = 5
        x_min, y_min = max(0, x_min-margem), max(0, y_min-margem)
        x_max, y_max = min(mask.shape[1], x_max+margem), min(mask.shape[0], y_max+margem)
        regiao = mask[y_min:y_max, x_min:x_max]
        regioes_unidas.append((x1+x_min, y1+y_min, x_max-x_min, y_max-y_min, regiao, contornos))

# ============================================================
# PREDICTING DIGITS (HANDLE 1 OR 2 DIGITS)
# ============================================================
nums_matriz = []

# ----------------------------
# PREDICTING DIGITS, MERGED, ALL CONFIDENCE
# ----------------------------
nums_matriz = []

for x, y, w, h, regiao, contornos in regioes_unidas:
    if regiao is None or len(contornos) == 0:
        nums_matriz.append(0)
        continue

    # Predict each contour in the cell
    cell_digits = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < 50:
            continue

        cx, cy, cw, ch = cv2.boundingRect(c)
        roi = regiao[cy:cy+ch, cx:cx+cw]

        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            continue

        label, conf = cnn_predict(roi)

        if conf < 0.7:
            continue

        cell_digits.append((cx, label))

    # Sort left to right
    # cell_digits = [(cx, label), ...]

    cell_digits.sort(key=lambda x: x[0])  # left → right
    digits = [d for _, d in cell_digits][:2]

    if len(digits) == 0:
        nums_matriz.append(-1)

    elif len(digits) == 1:
        nums_matriz.append(int(digits[0]))

    else:
        nums_matriz.append(int(f"{digits[0]}{digits[1]}"))



# Fill remaining empty cells to 16 entries
while len(nums_matriz) < 16:
    nums_matriz.append(-1)

# Convert to 4x4 numpy array
matrix_4x4 = np.array(nums_matriz).reshape(4, 4)
print("Final 4x4 matrix:\n", matrix_4x4)
