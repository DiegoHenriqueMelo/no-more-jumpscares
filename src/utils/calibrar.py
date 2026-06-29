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
                    print(f"\r  x={x:4d}, y={y:4d}   ", end="", flush=True)
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

def _capturar_janela_jogo() -> "np.ndarray | None":
    """Captura SÓ a janela do jogo — mesma região que o FNAFEnv usa para a
    observação e a detecção. Garante que a referência do vazio fique no mesmo
    enquadramento que o detector verá em produção."""
    import pygetwindow as gw
    from src.environment.fnaf_env import WINDOW_TITLE

    if not WINDOW_TITLE:
        print("FNAF_WINDOW_TITLE não configurado no .env — não dá para recortar a janela.")
        return None

    janelas = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not janelas:
        print(f"Janela '{WINDOW_TITLE}' não encontrada. O jogo está aberto?")
        return None

    win = janelas[0]
    regiao = {"left": win.left, "top": win.top, "width": win.width, "height": win.height}
    return cap.capturar_tela(regiao)


def capturar_vazio(regiao: str | None, variante: str | None = None):
    """Captura a referência do 'vazio' de uma região (porta ou câmera).

    Deixe a região VAZIA na tela antes de rodar:
      - porta_esq / porta_dir → no escritório, com a luz acesa no corredor vazio.
      - cam_*  → tab da câmera aberta, sala sem nenhum animatrônico.

    'variante' permite mais de um vazio por região — o detector compara contra a
    mais próxima. Ex.: a porta fechada mostra a sombra de forma diferente:
        python -m src.utils.calibrar vazio porta_esq            (porta aberta)
        python -m src.utils.calibrar vazio porta_esq fechada    (porta fechada)
    Para o balanço das câmeras, capture os dois extremos como variantes."""
    from src.environment.deteccao_visual import REGIOES, REFS_VAZIO, recortar_roi

    if regiao not in REGIOES:
        print(f"Região inválida: {regiao}")
        print(f"Use uma destas: {', '.join(REGIOES)}")
        return

    nome = regiao if not variante else f"{regiao}_{variante}"
    print(f"Região: {regiao}" + (f" | variante: {variante}" if variante else ""))
    print("Deixe a região VAZIA na tela. Você tem 5 segundos...")
    time.sleep(5)

    frame = _capturar_janela_jogo()
    if frame is None:
        return

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recorte = recortar_roi(cinza, regiao)

    REFS_VAZIO.mkdir(parents=True, exist_ok=True)
    caminho = REFS_VAZIO / f"{nome}.png"
    cv2.imwrite(str(caminho), recorte)
    print(f"Referência salva: {caminho}  ({recorte.shape[1]}x{recorte.shape[0]}px)")
    print("Confira no diagnóstico: python -m src.utils.testar_deteccao_animatronicos")


def selecionar_roi(regiao: str | None):
    """Define a ROI de uma região clicando 2 cantos NA TELA DO JOGO.

        python -m src.utils.calibrar roi janela_esq

    Clique o canto superior-esquerdo e depois o inferior-direito da área que você
    quer detectar (ex.: a janela, não a porta). Grava as frações em
    referencias/vazio/rois.json (override) — sem editar código.

    IMPORTANTE: depois de ajustar a ROI, RECAPTURE o vazio dela
    (calibrar vazio <regiao>), porque a referência antiga foi recortada com a ROI
    velha."""
    import sys
    import ctypes
    import pyautogui
    import pygetwindow as gw
    from src.environment.fnaf_env import WINDOW_TITLE
    from src.environment.deteccao_visual import REGIOES, salvar_override_roi

    if regiao not in REGIOES:
        print(f"Região inválida: {regiao}")
        print(f"Use uma destas: {', '.join(REGIOES)}")
        return

    if not WINDOW_TITLE:
        print("FNAF_WINDOW_TITLE não configurado no .env.")
        return
    janelas = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not janelas:
        print(f"Janela '{WINDOW_TITLE}' não encontrada. O jogo está aberto?")
        return
    win = janelas[0]

    if sys.platform == "win32":
        VK_LBUTTON = 0x01
        user32 = ctypes.windll.user32

        def _pressionado():
            return bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)

        def _clique(qual: str):
            print(f"Clique o canto {qual} da ROI de '{regiao}' na tela do jogo...")
            while _pressionado():
                time.sleep(0.01)
            while not _pressionado():
                time.sleep(0.01)
            pos = pyautogui.position()
            while _pressionado():
                time.sleep(0.01)
            print(f"  ({pos[0]}, {pos[1]})")
            return pos

        p1 = _clique("SUPERIOR-ESQUERDO")
        p2 = _clique("INFERIOR-DIREITO")
    else:
        input(f"Mouse no canto SUPERIOR-ESQUERDO da ROI de '{regiao}' e Enter: ")
        p1 = pyautogui.position()
        input("Agora no canto INFERIOR-DIREITO e Enter: ")
        p2 = pyautogui.position()

    x0, y0 = min(p1[0], p2[0]), min(p1[1], p2[1])
    x1, y1 = max(p1[0], p2[0]), max(p1[1], p2[1])

    def _frac(valor: int, origem: int, tamanho: int) -> float:
        return max(0.0, min(1.0, (valor - origem) / tamanho)) if tamanho else 0.0

    roi = (
        _frac(x0, win.left, win.width),
        _frac(y0, win.top,  win.height),
        _frac(x1, win.left, win.width),
        _frac(y1, win.top,  win.height),
    )
    caminho = salvar_override_roi(regiao, roi)
    print(f"\nROI de {regiao} salva: ({roi[0]:.3f}, {roi[1]:.3f}, {roi[2]:.3f}, {roi[3]:.3f})")
    print(f"  → {caminho}")
    print(f"Agora recapture o vazio:  python -m src.utils.calibrar vazio {regiao}")


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
    elif sys.argv[1] == "vazio":
        capturar_vazio(
            sys.argv[2] if len(sys.argv) > 2 else None,
            sys.argv[3] if len(sys.argv) > 3 else None,
        )
    elif sys.argv[1] == "roi":
        selecionar_roi(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(f"Argumento desconhecido: {sys.argv[1]}")
        print("Uso: python -m src.utils.calibrar [morte | vitoria | camera_aberta | roi <regiao> | vazio <regiao>]")
        print("  regiões: porta_esq, janela_esq, janela_dir, cam_1c, cam_2a, cam_2b, cam_4a, cam_4b")