from src.utils.capture import GameCapture
import cv2
import numpy as np

cap = GameCapture()

# Região apenas da barra verde de usage
regiao = {'left': 300, 'top': 840, 'width': 200, 'height': 15}
frame = cap.capturar_tela(regiao)
cv2.imwrite('debug_barra_proc.png', frame)

# Detecta pixels verdes
verde_mask = (
    (frame[:, :, 1].astype(int) - frame[:, :, 2].astype(int) > 50) &
    (frame[:, :, 1].astype(int) - frame[:, :, 0].astype(int) > 50) &
    (frame[:, :, 1] > 80)
)

pixels_verdes = np.sum(verde_mask)
print(f'Pixels verdes encontrados: {pixels_verdes}')
print(f'Total pixels: {verde_mask.size}')
print(f'Porcentagem verde: {pixels_verdes/verde_mask.size*100:.1f}%')