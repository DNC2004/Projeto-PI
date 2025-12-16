import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load main image / FUNCIONA!
image_path = 'imagem_final.png'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Could not load main image.")

img_teste = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite("imagem_teste.png", img_teste)


img_h, img_w = img.shape[:2]

# Extract digits via HSV mask (gold color)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite("imagem_teste_hsv.png", img_hsv)

#Mascaras Para deter os nums / FUNCIONA!
lower_gold = np.array([10, 60, 30], np.uint8)
upper_gold = np.array([150, 255, 255], np.uint8)
gold_mask = cv2.inRange(img_hsv, lower_gold, upper_gold)

kernel = np.ones((4, 4), np.uint8)

gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
gold_mask = cv2.dilate(gold_mask, kernel, iterations=1)

img_binary = gold_mask.copy()
cv2.imwrite("imagem_teste_binaria.png", img_binary)


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

# Contornos dos nums
contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

regiao_num = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # filter small noise
    if w < 20 or h < 20:
        continue

    regiao = img_binary[y:y+h, x:x+w]
    regiao_num.append((x, y, w, h, regiao))

regiao_num = sorted(regiao_num, key=lambda r: r[0])
# --- Merge nearby ROIs into multi-digit candidates ---
regioes_unidas = []
i = 0
while i < len(regiao_num):
    x, y, w, h, roi = regiao_num[i]

    # current merged bbox
    x_start = x
    x_end   = x + w
    y_top   = y
    y_bottom = y + h

    j = i + 1
    while j < len(regiao_num):
        x2, y2, w2, h2, roi2 = regiao_num[j]

        # horizontal gap between current merged box and next box
        gap = x2 - x_end

        # vertical overlap between current merged box and next box
        vertical_overlap = min(y_bottom, y2 + h2) - max(y_top, y2)

        # use current merged height and width, not only original w/h
        merged_w = x_end - x_start
        merged_h = y_bottom - y_top

        # be more permissive: allow up to 0.6 of digit width as gap,
        # and only require 30% vertical overlap
        if gap < max(merged_w, w2) * 0.7 and vertical_overlap > 0.2 * min(merged_h, h2):
            # expand current merged box
            x_end = max(x_end, x2 + w2)
            y_top = min(y_top, y2)
            y_bottom = max(y_bottom, y2 + h2)
            j += 1
        else:
            break

    regiao_unida = img_binary[y_top:y_bottom, x_start:x_end]
    regioes_unidas.append((x_start, y_top, x_end - x_start, y_bottom - y_top, regiao_unida))
    i = j

# Fazer Match aos templates
TARGET_SIZE = (60, 60)

def normalize(img):
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

detections = []

# Sort keys numerically descending, so 15,14,...,10 come first
sorted_templates = sorted(templates.items(),reverse=True)

for x, y, w, h, regiao in regioes_unidas:
    regiao_n = normalize(regiao)
    
    # Default
    best_score = -1
    best_label = None

    for label, temp in sorted_templates:
        temp_n = normalize(temp)

        score = cv2.matchTemplate(regiao_n, temp_n, cv2.TM_CCOEFF_NORMED)[0][0]
        
        # Encontra Match
        if score > best_score:
            best_score = score
            best_label = int(label)  # convert string to integer

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
        

# Non-maximum suppression globally over all detections
def nms_global(detections, overlapThresh=0.3):
    if not detections:
        return []
    boxes = np.array([[d["x"], d["y"], d["x"]+d["w"], d["y"]+d["h"], d["score"]] for d in detections], dtype=np.float32)
    idxs = np.argsort(boxes[:, 4])[::-1]

    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        suppress = [0]
        for pos in range(1, len(idxs)):
            j = idxs[pos]
            xx1 = max(boxes[i, 0], boxes[j, 0])
            yy1 = max(boxes[i, 1], boxes[j, 1])
            xx2 = min(boxes[i, 2], boxes[j, 2])
            yy2 = min(boxes[i, 3], boxes[j, 3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            area_j = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            if area_j == 0:
                continue
            overlap = inter / area_j
            if overlap > overlapThresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    final = []
    for i in pick:
        bx = boxes[i]
        d = detections[i]
        final.append({
            "x": int(bx[0]),
            "y": int(bx[1]),
            "w": int(bx[2] - bx[0]),
            "h": int(bx[3] - bx[1]),
            "label": d["label"],
            "score": d["score"]
        })
    return final

final_boxes = nms_global(detections, overlapThresh=0.3)
print(f"Detections after NMS: {len(final_boxes)}")

# Sort by reading order and build 4x4 matrix
final_boxes_sorted = sorted(final_boxes, key=lambda d: (d["y"], d["x"]))
labels = [int(d["label"]) for d in final_boxes_sorted]

while len(labels) < 16:
    labels.append(-1)
matrix_4x4 = np.array(labels[:16]).reshape(4, 4)


# Draw detections
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