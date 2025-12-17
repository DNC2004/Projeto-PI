import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import hashlib

# Load main image / FUNCIONA!
image_path = 'imagem_final.png'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Could not load main image.")

output_folder = "imagens_teste"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


img_teste = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
file_path = os.path.join(output_folder, "imagem_teste.png")
cv2.imwrite(file_path, img_teste)

img_h, img_w = img.shape[:2]

# Extract digits via HSV mask (gold color)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
file_path = os.path.join(output_folder, "imagem_teste_hsv.png")
cv2.imwrite(file_path, img_hsv)


#Mascaras Para deter os nums / FUNCIONA!
lower_gold = np.array([10, 60, 30], np.uint8)
upper_gold = np.array([150, 255, 255], np.uint8)
gold_mask = cv2.inRange(img_hsv, lower_gold, upper_gold)

kernel = np.ones((4, 4), np.uint8)

gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
gold_mask = cv2.dilate(gold_mask, kernel, iterations=1)

# Imagem 
img_binary = gold_mask.copy()
file_path = os.path.join(output_folder, "imagem_teste_binaria.png")
cv2.imwrite(file_path, img_binary)


# Load templates for digits 1-15 / FUNCIONA!
templates = {}
for i in range(1, 16):
    path = f"../templates/mask{i}.png"
    temp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if temp is None:
        print(f"Could not load template {path}")
        continue
    _, temp_bin = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp_bin = cv2.morphologyEx(temp_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    temp_bin = cv2.dilate(temp_bin, kernel, iterations=1)
    templates[str(i)] = temp_bin

print("Loaded templates:", templates.keys())


## A PARTIR DAQUI 

cell_h = img_binary.shape[0] // 4
cell_w = img_binary.shape[1] // 4

# Filtrar contornos válidos (remove ruído) 
contornos, hieraquia = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
contornos_validos = [c for c in contornos if cv2.contourArea(c) > 50]

# Contornos por cada número
# Juntar os contornos, quando existe mais do que um perto junta os dois
regioes_unidas = []
for row in range(4):
    for col in range(4):
        y1 = row * cell_h
        y2 = (row + 1) * cell_h
        x1 = col * cell_w
        x2 = (col + 1) * cell_w

        mascara_quadrado = img_binary[y1:y2, x1:x2]

        contornos, hieraquia = cv2.findContours(mascara_quadrado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contornos_validos = [c for c in contornos if cv2.contourArea(c) > 50]

        if not contornos_validos:
            continue

        # Quando temos dois contornos válidos no mesmo quadrado
        all_points = []
        for c in contornos_validos:
            all_points.extend(c.reshape(-1, 2))

        all_points = np.array(all_points)

        x_min = np.min(all_points[:, 0])
        y_min = np.min(all_points[:, 1])
        x_max = np.max(all_points[:, 0])
        y_max = np.max(all_points[:, 1])

        # Verficar se de facto é suposto serem um número com dois digitos
        margem = 5
        x_min = max(0, x_min - margem)
        y_min = max(0, y_min - margem)
        x_max = min(mascara_quadrado.shape[1], x_max + margem)
        y_max = min(mascara_quadrado.shape[0], y_max + margem)

        regiao = mascara_quadrado[y_min:y_max, x_min:x_max]

        # Juntar ao bolo de todos os quadrados
        regioes_unidas.append((
            x1 + x_min,
            y1 + y_min,
            x_max - x_min,
            y_max - y_min,
            regiao
        ))

    
def normalize(img, target_size=(60, 60)):
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

detections = []
sorted_templates = sorted(templates.items(), key=lambda x: int(x[0]))

# Debug
"""
for name in sorted_templates:
    print(f"Template Name: {name}")
"""
 
# Fazer match entre os templates e as regiões obtidas
for x, y, w, h, regiao in regioes_unidas:
    regiao_n = normalize(regiao)
    _, regiao_n = cv2.threshold(regiao_n, 128, 255, cv2.THRESH_BINARY)
    
    # Default
    best_score = -1
    best_label = None

    for label, temp in sorted_templates:
        temp_n = normalize(temp)
        _, temp_n = cv2.threshold(temp_n, 128, 255, cv2.THRESH_BINARY)

        score = cv2.matchTemplate(regiao_n, temp_n, cv2.TM_CCOEFF_NORMED)[0][0]
        
        
        # Debug nos match dos números
        """
        plt.figure(figsize=(4, 2))

        plt.subplot(1, 2, 1)
        plt.imshow(regiao_n, cmap='gray')
        plt.title("ROI normalizada")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(temp_n, cmap='gray')
        plt.title(f"Template {label}")
        plt.axis("off")

        plt.suptitle(f"Score = {score:.3f}")
        plt.tight_layout()
        plt.show()
        """
        
        
        # Encontra Match
        if score > best_score:
            best_score = score
            best_label = int(label) 

    # Guarda o melhor match
    if best_score > 0.6:
        detections.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "label": best_label,
            "score": best_score
            })
        

final_boxes = detections
print(f"DEBUG -- Total de deteções finais: {len(final_boxes)}")

# Sort by reading order and build 4x4 matrix
final_boxes_sorted = sorted(final_boxes, key=lambda d: (d["y"], d["x"]))

labels = [int(d["label"]) for d in final_boxes_sorted]

while len(labels) < 16:
    labels.append(-1)
    
matrix_4x4 = np.array(labels[:16]).reshape(4, 4)

# Desenhar as deteções na imagem
output = img.copy()
for d in final_boxes_sorted:
    x, y, w, h, label = d["x"], d["y"], d["w"], d["h"], d["label"]
    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(output, str(label), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15, 10))

plt.subplot(1,2,1)
plt.imshow(img_binary, cmap='gray')
plt.title("Nums Detetados")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(output_rgb)
plt.title("Num Detetados Imagem Real")
plt.axis("off")

plt.tight_layout()
plt.show()

print("4x4 matrix of detected numbers:\n", matrix_4x4)

# Guardar Imagem Output Final
imagens_finais_folder = "imagens_finais"
if not os.path.exists(imagens_finais_folder):
    os.makedirs(imagens_finais_folder)

# Ajuda a certificar de que não se guarda a mesma imagem duas vezes
img_hash = hashlib.md5(output_rgb).hexdigest()

file_path_output = os.path.join(imagens_finais_folder, f"imagem_final_{img_hash}.png")
if os.path.exists(file_path_output):
    print("DEBUG -- IMAGEM JÁ GUARDADA")
    
else:
    cv2.imwrite(file_path_output, output_rgb)
    
print(f"DEBUG -- Imagem final guardada em: {file_path_output}")

