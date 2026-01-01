import subprocess
import sys
import os
import cv2
import numpy as np
import random
from uuid import uuid4

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
# TEMPLATE DATASET GENERATION (only if not exists)
# ============================================================
if not os.path.exists("dataset/train"):
    print("DEBUG -- Generating dataset...")

    templates = {}
    for i in range(1, 16):
        path = f"../templates/mask{i}.png"
        img_t = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, img_t = cv2.threshold(img_t, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        templates[i] = img_t

    def augment(img):
        scale = random.uniform(0.8, 1.2)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=0)
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=0)
        if random.random() < 0.5:
            img = cv2.erode(img, np.ones((2,2), np.uint8))
        else:
            img = cv2.dilate(img, np.ones((2,2), np.uint8))
        img = cv2.resize(img, (60, 60))
        return img

    os.makedirs("dataset", exist_ok=True)

    # Single-digit dataset
    for label, template in templates.items():
        class_dir = f"dataset/{label}"
        os.makedirs(class_dir, exist_ok=True)
        for _ in range(1000):
            cv2.imwrite(f"{class_dir}/{uuid4().hex}.png", augment(template))

    # Empty class
    empty_dir = "dataset/0"
    os.makedirs(empty_dir, exist_ok=True)
    for _ in range(1500):
        noise = np.zeros((60, 60), dtype=np.uint8)
        cv2.imwrite(f"{empty_dir}/{uuid4().hex}.png", noise)

    # Split train/val
    for label in os.listdir("dataset"):
        imgs = os.listdir(f"dataset/{label}")
        train, val = train_test_split(imgs, test_size=0.2)
        for split, files in [("train", train), ("val", val)]:
            os.makedirs(f"dataset/{split}/{label}", exist_ok=True)
            for f in files:
                shutil.move(f"dataset/{label}/{f}", f"dataset/{split}/{label}/{f}")
else:
    print("DEBUG -- Dataset already exists, skipping generation")

# ============================================================
# CNN PREPROCESSING
# ============================================================
def preprocess_for_cnn(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.resize(img, (60, 60))
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)

class CNNTransform:
    def __call__(self, img):
        img = np.array(img)
        return preprocess_for_cnn(img)

transform = CNNTransform()

train_ds = datasets.ImageFolder("dataset/train", transform=transform)
val_ds   = datasets.ImageFolder("dataset/val", transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

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
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 16)  # 0=empty, 1–15=digits
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
    SKIP_TRAINING = True
else:
    SKIP_TRAINING = False

if not SKIP_TRAINING:
    for epoch in range(6):  # fewer epochs for CPU
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
        probs = F.softmax(model(tensor), dim=1)[0]
        label = probs.argmax().item()
        confidence = probs[label].item()
    if label == 0:
        return -1, confidence
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
        nums_matriz.append(-1)
        continue

    # Predict each contour in the cell
    cell_digits = []
    for c in contornos:
        cx, cy, cw, ch = cv2.boundingRect(c)
        roi = regiao[cy:cy+ch, cx:cx+cw]
        label, conf = cnn_predict(roi)
        print(f"DEBUG -- Conf: {conf} | Label: {label}")
        if conf < 0.4:
            label = -1
        cell_digits.append(label if label != -1 else 0)  # 0 for empty

    # Sort left to right
    cell_digits.sort()

    # Merge digits if more than one
    if len(cell_digits) == 1:
        nums_matriz.append(cell_digits[0])
    else:
        merged = int("".join(str(d) for d in cell_digits))
        nums_matriz.append(merged)

# Fill remaining empty cells to 16 entries
while len(nums_matriz) < 16:
    nums_matriz.append(-1)

# Convert to 4x4 numpy array
matrix_4x4 = np.array(nums_matriz).reshape(4, 4)
print("Final 4x4 matrix:\n", matrix_4x4)
