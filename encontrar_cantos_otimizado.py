import numpy as np
import matplotlib.pyplot as plt
import cv2

# Encontra o tabuleiro de jogo na fotografia
caminho_img = 'Imagens/jogo_stor.jpg' # Não funciona
#caminho_img = 'Imagens/jogo_distorced.jpg' ## ATUAL / Funciona Tudo
#caminho_img = 'Imagens/jogo_perp.jpeg' # Funciona Tudo
#caminho_img = 'Imagens/jogo_blackgorund.jpg' # Funciona Tudo
#caminho_img = 'Imagens/jogo_scuro.jpg'  # Funciona Tudo
#caminho_img = 'Imagens/jogo_purp.jpg' # Funciona Tudo
#caminho_img = 'Imagens/jogo_red.jpg' # Funciona Tudo


img = cv2.imread(caminho_img)

#Preparar Imagem
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
escala = 1200 / max(img_rgb.shape[:2])
img_rgb = cv2.resize(img_rgb, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

# Converter para HSV
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

# --- Segmentar vermelho ---
mask_r1 = cv2.inRange(img_hsv, (0, 80, 60), (10, 255, 255))
mask_r2 = cv2.inRange(img_hsv, (170, 80, 60), (180, 255, 255))
mask_red = cv2.bitwise_or(mask_r1, mask_r2)

# --- Segmentar branco ---
mask_white = cv2.inRange(img_hsv, (0, 0, 160), (180, 80, 255))

mask_red = cv2.medianBlur(mask_red, 5)
mask_white = cv2.medianBlur(mask_white, 5)

ker = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

mask_r_d = cv2.dilate(mask_red, ker, iterations = 1)
mask_w_d = cv2.dilate(mask_white, ker, iterations = 1)

inter = cv2.bitwise_and(mask_r_d, mask_w_d)

num_r, labels_r = cv2.connectedComponents(mask_r_d, connectivity = 8)

num_w, labels_w = cv2.connectedComponents(mask_w_d)
 
# Perceber melhor
ids_r = np.unique(labels_r[inter > 0])
ids_w = np.unique(labels_w[inter > 0])
 
# Remover fundo (0)
ids_r = ids_r[ids_r != 0]
ids_w = ids_w[ids_w != 0]

# Máscara nos compontentes que se toquem 
mask_r_touch = np.zeros_like(labels_r, dtype=np.uint8)
for rid in ids_r: 
    mask_r_touch |= (labels_r == rid)
mask_r_touch = (mask_r_touch * 255).astype(np.uint8)
 
mask_w_touch = np.zeros_like(labels_w, dtype=np.uint8)
for wid in ids_w: 
    mask_w_touch |= (labels_w == wid)
mask_w_touch = (mask_w_touch * 255).astype(np.uint8)
 
# Unir as máscaras das zonas brancas e vermelhas
mask_tiles_touch = cv2.bitwise_or(mask_r_touch, mask_w_touch)
 
mask = cv2.erode(mask_tiles_touch, ker, iterations=1)
 
# Contornos
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise ValueError("Nenhum contorno encontrado.")
 
largest = max(contours, key=cv2.contourArea)
perimetro = cv2.arcLength(largest, True)
approx = cv2.approxPolyDP(largest, 0.01 * perimetro, True)
 
mask_refinada = np.zeros_like(mask)
cv2.drawContours(mask_refinada, [approx], -1, 255, thickness=-1)

#
contours, hierarchy = cv2.findContours(mask_refinada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)

epsilon = 0.01 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)
points = approx.reshape(-1, 2)

segments = []
for i in range(len(points)):
    p1 = points[i]
    p2 = points[(i + 1) % len(points)]
    length = np.linalg.norm(p1 - p2)
    segments.append((length, p1, p2))

# Escolhe os 4 melhores contornos -- Área possível do tabuleiro 
segments.sort(key=lambda x: x[0], reverse=True)
top4 = segments[:4]

# Delimitar o tabuleiro em si
def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return np.array([px, py])

# Gerar as interceções
intersections = []
for i in range(len(top4)):
    for j in range(i+1, len(top4)):
        inter = line_intersection(top4[i][1], top4[i][2], top4[j][1], top4[j][2])
        if inter is not None:
            intersections.append(inter)

# Encontrar os centroide 
M = cv2.moments(approx)
cx = int(M["m10"] / M["m00"])
cy = int(M["m01"] / M["m00"])
centroid = np.array([cx, cy])

# Escolher as interseções mais perto
if len(intersections) > 4:
    intersections = sorted(intersections, key=lambda p: np.linalg.norm(p - centroid))[:4]

overlay = img_rgb.copy()

for pt in intersections:
    cv2.circle(overlay, tuple(np.int32(pt)), 6, (0,255,0), -1)

cv2.circle(overlay, (cx, cy), 6, (255,0,0), -1)  # centroid
for (_, p1, p2) in top4:
    cv2.line(overlay, tuple(p1), tuple(p2), (0,0,255), 2)

alpha = 0.6
output = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # Top-Left
    rect[2] = pts[np.argmax(s)]   # Bottom-Right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left

    return rect

# Converter lista de interseções para matriz e ordenar
corners = np.array(intersections, dtype="float32")
rect = order_points(corners)

# Perpestiva final
dst = np.array([
    [100, 100],
    [600, 100],
    [600, 600],
    [100, 600]
], dtype="float32")

M = cv2.getPerspectiveTransform(rect, dst)

mask_3c = cv2.merge([mask_refinada]*3)
masked_img = cv2.bitwise_and(img_rgb, mask_3c)

x_min = int(np.min(rect[:, 0]))
x_max = int(np.max(rect[:, 0]))
y_min = int(np.min(rect[:, 1]))
y_max = int(np.max(rect[:, 1]))

cropped_masked = masked_img[y_min:y_max, x_min:x_max]

size = max(y_max - y_min, x_max - x_min)
dst_tight = np.array([
    [0, 0],
    [size, 0],
    [size, size],
    [0, size]
], dtype="float32")

rect_cropped = rect.copy()
rect_cropped[:, 0] -= x_min
rect_cropped[:, 1] -= y_min

M_tight = cv2.getPerspectiveTransform(rect_cropped, dst_tight)
warped_tight = cv2.warpPerspective(cropped_masked, M_tight, (size, size))
warped_export = cv2.cvtColor(warped_tight, cv2.COLOR_BGR2RGB)

cv2.imwrite("imagem_final.png", warped_export)

# Plots com o processo até chegar à imagem final  
plt.figure(figsize=(20, 12))

plt.subplot(2, 4, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(mask_red, cmap="gray")
plt.title("Máscara Vermelha")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(mask_white, cmap="gray")
plt.title("Máscara Branca")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(mask_tiles_touch)
plt.title("Área de Jogo")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(mask_refinada)
plt.title("Área de jogo rough")
plt.axis("off")

plt.subplot(2, 4, 6)
# Overlay with intersections
overlay = img_rgb.copy()
for pt in intersections:
    cv2.circle(overlay, tuple(np.int32(pt)), 6, (0,255,0), -1)
cv2.circle(overlay, (cx, cy), 6, (255,0,0), -1)

for (_, p1, p2) in top4:
    cv2.line(overlay, tuple(p1), tuple(p2), (0,0,255), 2)

alpha = 0.6
output = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)
plt.imshow(output)
plt.title("Overlay com interseções")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(masked_img)
plt.title("Imagem Original Apenas Área Detectada")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(warped_tight)
plt.title("Imagem Final Warp sem Bordas")
plt.axis("off")

plt.tight_layout()
plt.show()
