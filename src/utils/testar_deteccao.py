"""
Diagnóstico de detecção de morte/vitória em tempo real.
Execute, vá para o jogo, provoque morte ou vitória e observe os scores.
Ctrl+C para sair.
"""
import cv2
import time
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.capture import GameCapture
from src.environment.fnaf_env import WINDOW_TITLE
import pygetwindow as gw


REFS = Path(__file__).parent / "referencias"
REF_SIZE = (1280, 720)


def carregar_templates():
    morte_img   = cv2.imread(str(REFS / "morte.jpg"),   cv2.IMREAD_GRAYSCALE)
    vitoria_img = cv2.imread(str(REFS / "vitoria.png"), cv2.IMREAD_GRAYSCALE)

    if morte_img is None or vitoria_img is None:
        raise FileNotFoundError(
            "Templates não encontrados em src/utils/referencias/. "
            "Execute: python -m src.utils.calibrar morte  e  python -m src.utils.calibrar vitoria"
        )

    h, w = morte_img.shape
    tmpl_morte = morte_img[int(h * 0.88):, int(w * 0.82):]

    h, w = vitoria_img.shape
    tmpl_vitoria = vitoria_img[int(h * 0.38):int(h * 0.58), int(w * 0.38):int(w * 0.62)]

    print(f"Template morte:   {tmpl_morte.shape}  (h x w)")
    print(f"Template vitoria: {tmpl_vitoria.shape}  (h x w)")
    return tmpl_morte, tmpl_vitoria


def capturar_janela(cap: GameCapture, titulo: str) -> np.ndarray:
    janelas = gw.getWindowsWithTitle(titulo)
    if janelas:
        win = janelas[0]
        regiao = {"left": win.left, "top": win.top,
                  "width": win.width, "height": win.height}
        frame = cap.capturar_tela(regiao)
    else:
        frame = cap.capturar_tela()
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(cinza, REF_SIZE)


def score(frame: np.ndarray, template: np.ndarray) -> float:
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


if __name__ == "__main__":
    cap = GameCapture()
    tmpl_morte, tmpl_vitoria = carregar_templates()
    titulo = WINDOW_TITLE or "Five Nights at Freddy's"
    print(f"\nMonitorando janela: '{titulo}'")
    print("Ctrl+C para sair\n")
    print(f"{'Morte score':>15}  {'Vitória score':>15}  {'Status':>25}")
    print("-" * 60)

    while True:
        frame = capturar_janela(cap, titulo)
        s_morte   = score(frame, tmpl_morte)
        s_vitoria = score(frame, tmpl_vitoria)

        if s_morte > 0.70:
            status = "<<< MORTE DETECTADA >>>"
            cv2.imwrite("debug_deteccao_morte.png", frame)
        elif s_vitoria > 0.70:
            status = "<<< VITORIA DETECTADA >>>"
            cv2.imwrite("debug_deteccao_vitoria.png", frame)
        else:
            status = "normal"

        print(f"{s_morte:15.3f}  {s_vitoria:15.3f}  {status:>25}", end="\r")
        time.sleep(0.3)
