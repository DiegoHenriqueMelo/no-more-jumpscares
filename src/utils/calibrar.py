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

def capturar_camera_aberta():
    """
    Com a câmera ABERTA no jogo (qualquer tab), clique sobre o indicador 'YOU'
    no mapa de câmeras. Salva o recorte em referencias/camera_aberta.png.
    """
    import ctypes
    import sys
    import pyautogui

    VK_LBUTTON = 0x01

    print("Abra a câmera no FNAF1 (qualquer tab serve).")
    print("Localize o quadrado 'YOU' no mapa de câmeras (indica sua posição no mapa).")
    print("Clique sobre ele para capturar o template.")
    print("Pressione Ctrl+C para cancelar.\n")

    MARG_X, MARG_Y = 40, 30  # recorte: 80×60 px ao redor do clique

    if sys.platform == "win32":
        user32 = ctypes.windll.user32

        def _pressionado():
            return bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)

        while _pressionado():
            time.sleep(0.01)

        ultimo = (-1, -1)
        try:
            while True:
                x, y = pyautogui.position()
                if (x, y) != ultimo:
                    frame = cap.capturar_tela()
                    h, w = frame.shape[:2]
                    if 0 <= y < h and 0 <= x < w:
                        b, g, r = int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2])
                        print(f"\r  x={x:4d}, y={y:4d} | R={r:3d} G={g:3d} B={b:3d}   ", end="", flush=True)
                    ultimo = (x, y)

                if _pressionado():
                    cx, cy = pyautogui.position()
                    while _pressionado():
                        time.sleep(0.01)

                    frame = cap.capturar_tela()
                    h, w = frame.shape[:2]
                    x1 = max(0, cx - MARG_X)
                    y1 = max(0, cy - MARG_Y)
                    x2 = min(w, cx + MARG_X)
                    y2 = min(h, cy + MARG_Y)

                    recorte = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    caminho = "src/utils/referencias/camera_aberta.png"
                    cv2.imwrite(caminho, recorte)
                    print(f"\nTemplate salvo: {caminho}")
                    print(f"Tamanho: {recorte.shape[1]}x{recorte.shape[0]}px | Centro capturado: ({cx}, {cy})")
                    return

                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nCancelado.")
    else:
        input("Posicione o mouse sobre o 'YOU' e pressione Enter: ")
        cx, cy = pyautogui.position()
        frame = cap.capturar_tela()
        h, w = frame.shape[:2]
        x1 = max(0, cx - MARG_X)
        y1 = max(0, cy - MARG_Y)
        x2 = min(w, cx + MARG_X)
        y2 = min(h, cy + MARG_Y)
        recorte = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        cv2.imwrite("src/utils/referencias/camera_aberta.png", recorte)
        print("Template salvo em src/utils/referencias/camera_aberta.png!")

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
    elif sys.argv[1] == "camera_aberta":
        capturar_camera_aberta()
    else:
        print(f"Argumento desconhecido: {sys.argv[1]}")
        print("Uso: python -m src.utils.calibrar [morte | vitoria | camera_aberta]")