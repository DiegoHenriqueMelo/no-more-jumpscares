import cv2
import time
from src.utils.capture import GameCapture

"""
Script de calibragem — roda uma vez para capturar as imagens de referência.
Execute cada função na hora certa:
  - capturar_morte()   → rode quando a tela de game over aparecer
  - capturar_vitoria() → rode quando o "6 AM" aparecer
  - capturar_coords()  → mostra as coordenadas do mouse em tempo real
"""

cap = GameCapture()

def capturar_morte():
    """
    Abra o FNAF1, morra de propósito, e rode essa função
    ENQUANTO a tela de game over estiver aparecendo.
    """
    print("Você tem 5 segundos para deixar a tela de game over aparecer...")
    time.sleep(5)

    frame = cap.capturar_tela()
    cv2.imwrite("src/utils/referencias/morte.png", frame)
    print("Imagem de morte salva!")

def capturar_vitoria():
    """
    Rode quando o '6 AM' aparecer na tela.
    """
    print("Você tem 5 segundos para deixar o 6 AM aparecer...")
    time.sleep(5)

    frame = cap.capturar_tela()
    cv2.imwrite("src/utils/referencias/vitoria.png", frame)
    print("Imagem de vitória salva!")

def capturar_coords():
    import pyautogui
    print("Movendo o mouse sobre os botões do jogo...")
    print("Pressione Ctrl+C para parar.\n")

    ultimo_x, ultimo_y = 0, 0
    try:
        while True:
            x, y = pyautogui.position()
            if x != ultimo_x or ultimo_y != y:
                print(f"x={x}, y={y}")
                ultimo_x, ultimo_y = x, y
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nPronto!")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        capturar_coords()
    elif sys.argv[1] == "morte":
        capturar_morte()
    elif sys.argv[1] == "vitoria":
        capturar_vitoria()