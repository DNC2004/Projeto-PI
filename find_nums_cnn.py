import subprocess
import sys
import os
import cv2
import numpy as np
from torchvision import transforms
import torchvision.datasets as datasets


try:
    import torch 
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

# Transformar a imagem em binária -- Igual ao template matching
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

""" 1. PRÉ-PROCESSAMENTO """

# Base de dados MNIST
mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST( root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


def preprocess_for_cnn(img):
    img = cv2.resize(img, (28, 28))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5

    return torch.from_numpy(img).unsqueeze(0)


""" 2. MODELO CNN"""

class DigitCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(DigitCNN, self).__init__()
        # 1ª Layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        # Layer Max Pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2ª Layer
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


""" 3. TREINAR O MODELO """

MODEL_PATH = "digit_cnn.pth"
if os.path.exists(MODEL_PATH):
    print("DEBUG -- Loading trained CNN...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    for x, y in train_loader:
        out = model(x)             
        print("DEBUG -- Predicted classes:", out.argmax(1)) 
        break 
    SKIP_TRAINING = True
else:
    SKIP_TRAINING = False

if not SKIP_TRAINING:
    for epoch in range(10): 
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

""" 4. PREVISÕES CNN """

def cnn_predict(roi):
    tensor = preprocess_for_cnn(roi).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    label = probs.argmax().item()
    confidence = probs[label].item()

    return label, confidence

# Handling de dois dígitos no mesmo quadrado
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

        pts = np.vstack([c.reshape(-1,2) for c in contornos])
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        margem = 5
        x_min, y_min = max(0, x_min-margem), max(0, y_min-margem)
        x_max, y_max = min(mask.shape[1], x_max+margem), min(mask.shape[0], y_max+margem)
        regiao = mask[y_min:y_max, x_min:x_max]
        regioes_unidas.append((x1+x_min, y1+y_min, x_max-x_min, y_max-y_min, regiao, contornos))

nums_matriz = []

for x, y, w, h, regiao, contornos in regioes_unidas:
    if regiao is None or len(contornos) == 0:
        nums_matriz.append(0)
        continue

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

    cell_digits.sort(key=lambda x: x[0]) 
    digits = [d for _, d in cell_digits][:2]

    if len(digits) == 0:
        nums_matriz.append(0)

    elif len(digits) == 1:
        nums_matriz.append(int(digits[0]))

    else:
        nums_matriz.append(int(f"{digits[0]}{digits[1]}"))


while len(nums_matriz) < 16:
    nums_matriz.append(0)

# Convert to 4x4 numpy array
matrix_4x4 = np.array(nums_matriz).reshape(4, 4)
print("Tabuleiro de jogo da fotografia:\n", matrix_4x4)
