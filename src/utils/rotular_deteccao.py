"""
Coleta amostras ROTULADAS para medir a PRECISÃO da detecção (sem confiar no olho).

Uso:
    python -m src.utils.rotular_deteccao <regiao> <vazio|cheio> [n]

Deixe a cena pronta NA CONDIÇÃO OBSERVÁVEL (luz daquele lado acesa / câmera aberta):
    - vazio: a região SEM animatrônico
    - cheio: a região COM o animatrônico (ou a sombra, no caso da janela)

Tem 4s de contagem para você focar o jogo; captura n frames (padrão 5) com leve
intervalo, pegando variação de pose/balanço. Repita várias vezes alternando
vazio/cheio, em situações diferentes. Depois rode:

    python -m src.utils.avaliar_deteccao

As amostras vão para dados/rotulos_deteccao/<regiao>/ (gitignored).
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.capture import GameCapture
from src.environment.deteccao_visual import REGIOES

DEST = Path("dados/rotulos_deteccao")


def _capturar_janela(cap: GameCapture):
    import pygetwindow as gw
    from src.environment.fnaf_env import WINDOW_TITLE

    if not WINDOW_TITLE:
        print("FNAF_WINDOW_TITLE não configurado no .env.")
        return None
    janelas = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not janelas:
        print(f"Janela '{WINDOW_TITLE}' não encontrada. O jogo está aberto?")
        return None
    w = janelas[0]
    return cap.capturar_tela({"left": w.left, "top": w.top, "width": w.width, "height": w.height})


def main():
    if (len(sys.argv) < 3 or sys.argv[1] not in REGIOES
            or sys.argv[2] not in ("vazio", "cheio")):
        print("Uso: python -m src.utils.rotular_deteccao <regiao> <vazio|cheio> [n]")
        print("regiões:", ", ".join(REGIOES))
        return

    regiao, rotulo = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    cap = GameCapture()
    pasta = DEST / regiao
    pasta.mkdir(parents=True, exist_ok=True)

    print(f"[{regiao} / {rotulo}] deixe a cena pronta (condição observável).")
    print(f"Capturando {n} frames em 4s...")
    time.sleep(4)

    salvos = 0
    for _ in range(n):
        frame = _capturar_janela(cap)
        if frame is None:
            break
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(str(pasta / f"{rotulo}_{ts}.png"), frame)
        salvos += 1
        time.sleep(0.4)

    n_vazio = len(list(pasta.glob("vazio_*.png")))
    n_cheio = len(list(pasta.glob("cheio_*.png")))
    print(f"Salvos {salvos} frames. Total de '{regiao}': vazio={n_vazio}, cheio={n_cheio}")
    print("Junte pelo menos ~10 de cada (em situações variadas) antes de avaliar.")


if __name__ == "__main__":
    main()
