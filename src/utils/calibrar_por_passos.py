"""
Calibracao guiada por passos para coordenadas de clique do FNAF.

Mantem o jeito atual (src.utils.calibrar), mas oferece um fluxo guiado:
1) botao de iniciar/reset (menu)
2) portas e luzes
3) abrir/fechar camera (um unico botao)
4) todas as cameras

No final, exibe o bloco formatado para colar no .env.
"""

from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass

import pyautogui


VK_LBUTTON = 0x01


@dataclass(frozen=True)
class PassoCalibracao:
    chave_env: str
    descricao: str
    prefixo_env: str = "FNAF_COORD"

    @property
    def variavel_base(self) -> str:
        return f"{self.prefixo_env}_{self.chave_env}"


GRUPOS: list[tuple[str, list[PassoCalibracao]]] = [
    (
        "Menu inicial",
        [
            PassoCalibracao(
                "RESET_CLICK",
                "Botao de iniciar/continuar no menu",
                prefixo_env="FNAF",
            ),
        ],
    ),
    (
        "Portas e luzes",
        [
            PassoCalibracao("PORTA_ESQUERDA", "Botao da porta esquerda"),
            PassoCalibracao("PORTA_DIREITA", "Botao da porta direita"),
            PassoCalibracao("LUZ_ESQUERDA", "Botao da luz esquerda"),
            PassoCalibracao("LUZ_DIREITA", "Botao da luz direita"),
        ],
    ),
    (
        "Controle de camera",
        [
            PassoCalibracao("ABRIR_FECHAR_CAMERA", "Botao abrir/fechar camera"),
        ],
    ),
    (
        "Cameras",
        [
            PassoCalibracao("CAMERA_1A", "Camera 1A"),
            PassoCalibracao("CAMERA_1B", "Camera 1B"),
            PassoCalibracao("CAMERA_1C", "Camera 1C"),
            PassoCalibracao("CAMERA_2A", "Camera 2A"),
            PassoCalibracao("CAMERA_2B", "Camera 2B"),
            PassoCalibracao("CAMERA_3", "Camera 3"),
            PassoCalibracao("CAMERA_4A", "Camera 4A"),
            PassoCalibracao("CAMERA_4B", "Camera 4B"),
            PassoCalibracao("CAMERA_5", "Camera 5"),
            PassoCalibracao("CAMERA_6", "Camera 6"),
            PassoCalibracao("CAMERA_7", "Camera 7"),
        ],
    ),
]


def _esperar_clique_esquerdo_windows(intervalo: float = 0.01) -> tuple[int, int]:
    """Aguarda o proximo clique esquerdo global e retorna (x, y)."""
    user32 = ctypes.windll.user32

    def botao_pressionado() -> bool:
        return bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)

    # Evita capturar o clique anterior caso o botao ainda esteja pressionado.
    while botao_pressionado():
        time.sleep(intervalo)

    while True:
        if botao_pressionado():
            x, y = pyautogui.position()

            # Aguarda soltar para evitar leitura dupla no proximo passo.
            while botao_pressionado():
                time.sleep(intervalo)

            return int(x), int(y)
        time.sleep(intervalo)


def _capturar_coordenada() -> tuple[int, int]:
    """
    Captura por clique no Windows.
    Em outros sistemas, faz fallback para Enter na posicao atual do mouse.
    """
    if sys.platform == "win32":
        return _esperar_clique_esquerdo_windows()

    input("Posicione o mouse e pressione Enter para capturar: ")
    x, y = pyautogui.position()
    return int(x), int(y)


def _capturar_pixel_colorido() -> tuple[int, int]:
    """Exibe cor do pixel em tempo real e captura na posição do clique."""
    import os
    _raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _raiz not in sys.path:
        sys.path.insert(0, _raiz)
    from src.utils.capture import GameCapture
    cap = GameCapture()

    if sys.platform == "win32":
        user32 = ctypes.windll.user32

        def _pressionado() -> bool:
            return bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)

        while _pressionado():
            time.sleep(0.01)

        ultimo: tuple[int, int] = (-1, -1)
        while True:
            x, y = pyautogui.position()
            if (x, y) != ultimo:
                frame = cap.capturar_tela()
                h, w = frame.shape[:2]
                if 0 <= y < h and 0 <= x < w:
                    b_val, g_val, r_val = int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2])
                    print(
                        f"\r  x={x:4d}, y={y:4d} | R={r_val:3d} G={g_val:3d} B={b_val:3d}   ",
                        end="",
                        flush=True,
                    )
                ultimo = (x, y)
            if _pressionado():
                x, y = pyautogui.position()
                while _pressionado():
                    time.sleep(0.01)
                print()
                return int(x), int(y)
            time.sleep(0.05)
    else:
        input("Posicione o mouse na luz vermelha e pressione Enter: ")
        x, y = pyautogui.position()
        return int(x), int(y)


def _imprimir_bloco_env(coords: dict[str, tuple[int, int]]) -> None:
    print("\n" + "=" * 72)
    print("BLOCO FORMATADO PARA O .env")
    print("=" * 72)
    print("# Coordenadas de reset e das acoes do agente")
    print()

    for _, passos in GRUPOS:
        for passo in passos:
            x, y = coords[passo.variavel_base]
            print(f"{passo.variavel_base}_X={x}")
            print(f"{passo.variavel_base}_Y={y}")
            print()

    if "FNAF_CAMERA_PIXEL" in coords:
        px, py = coords["FNAF_CAMERA_PIXEL"]
        print("# Pixel de verificacao de camera (luz vermelha piscando)")
        print(f"FNAF_CAMERA_PIXEL_X={px}")
        print(f"FNAF_CAMERA_PIXEL_Y={py}")
        print()


def executar_calibracao_guiada() -> None:
    print("Calibracao guiada por passos")
    print("Abra o jogo em modo janela e mantenha a escala do Windows em 100%.")
    print("Para cada passo, clique COM O BOTAO ESQUERDO no ponto indicado.")
    print("Pressione Ctrl+C para cancelar a qualquer momento.")
    input("\nPressione Enter para iniciar... ")

    total_coords = sum(len(passos) for _, passos in GRUPOS)
    total_passos = total_coords + 1  # +1 para o pixel de câmera
    atual = 1
    coords: dict[str, tuple[int, int]] = {}

    for nome_grupo, passos in GRUPOS:
        print("\n" + "-" * 72)
        print(f"Etapa: {nome_grupo}")
        print("-" * 72)

        for passo in passos:
            print(f"[{atual}/{total_passos}] Clique agora em: {passo.descricao}")
            x, y = _capturar_coordenada()
            coords[passo.variavel_base] = (x, y)
            print(f"    Capturado -> x={x}, y={y}")
            atual += 1

    print("\n" + "-" * 72)
    print("Etapa: Verificacao de estado da camera")
    print("-" * 72)
    print(f"[{atual}/{total_passos}] Abra a CAMERA no jogo e mova o mouse sobre")
    print("  a luz vermelha piscando. Clique quando R for alto e G/B baixos.")
    px, py = _capturar_pixel_colorido()
    coords["FNAF_CAMERA_PIXEL"] = (px, py)
    print(f"    Capturado -> x={px}, y={py}")

    _imprimir_bloco_env(coords)


if __name__ == "__main__":
    try:
        executar_calibracao_guiada()
    except KeyboardInterrupt:
        print("\nCalibracao cancelada pelo usuario.")