import cv2
import json
import os
import time
import ctypes
from datetime import datetime
from pathlib import Path

import keyboard
import pyautogui
import pygetwindow as gw

from src.utils.capture import GameCapture
from src.environment.fnaf_env import COORDS, ACOES

cap = GameCapture()

# ─── Mapeamento tecla → ação ──────────────────────────────────────
TECLAS = {
    "a":   "porta_esquerda",
    "d":   "porta_direita",
    "q":   "luz_esquerda",
    "e":   "luz_direita",
    "tab": "abrir_fechar_camera",
    "1":   "camera_1a",
    "2":   "camera_1b",
    "3":   "camera_1c",
    "4":   "camera_2a",
    "5":   "camera_2b",
    "6":   "camera_3",
    "7":   "camera_4a",
    "8":   "camera_4b",
    "9":   "camera_5",
    "0":   "camera_6",
    "-":   "camera_7",
}

def acao_para_numero(nome_acao: str) -> int:
    for numero, nome in ACOES.items():
        if nome == nome_acao:
            return numero
    return 0

def executar_acao_no_jogo(nome_acao: str):
    """Executa a ação no jogo — clique simples ou arrasto para câmera."""
    if nome_acao == "abrir_fechar_camera":
        # Câmera precisa de clique e arrasto para cima
        x, y = COORDS["abrir_fechar_camera"]
        pyautogui.mouseDown(x, y)
        time.sleep(0.05)
        pyautogui.moveTo(x, y - 200, duration=0.2)  # arrasta para cima
        pyautogui.mouseUp()

    elif nome_acao in COORDS:
        x, y = COORDS[nome_acao]
        pyautogui.click(x, y)

def gravar():
    pasta = f"gameplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(f"dados/{pasta}/frames", exist_ok=True)

    print("=== Gravação iniciada! ===")
    print("Mapeamento de teclas:")
    for tecla, acao in TECLAS.items():
        print(f"  [{tecla}] → {acao}")
    print("\n[F10] → Para a gravação\n")
    print("Você tem 3 segundos para focar o jogo...")
    time.sleep(3)

    dados      = []
    frame_idx  = 0
    acao_atual = "nada"

    # Registra handlers para cada tecla
    def fazer_handler(nome_acao):
        def handler(event):
            nonlocal acao_atual
            acao_atual = nome_acao
            executar_acao_no_jogo(nome_acao)
            print(f"  [{event.name}] → {nome_acao}")
        return handler

    hooks = []
    for tecla, nome_acao in TECLAS.items():
        h = keyboard.on_press_key(tecla, fazer_handler(nome_acao))
        hooks.append(h)

    try:
        while not keyboard.is_pressed("f10"):
            # Captura janela do jogo
            janelas = gw.getWindowsWithTitle("Five Nights at Freddy's")
            if not janelas:
                print("Jogo não encontrado! Aguardando...")
                time.sleep(1)
                continue

            win = janelas[0]
            regiao = {
                "left":   win.left,
                "top":    win.top,
                "width":  win.width,
                "height": win.height,
            }
            frame = cap.capturar_tela(regiao)

            # Processa frame
            frame_cinza   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_pequeno = cv2.resize(frame_cinza, (84, 84))

            # Salva frame
            caminho_frame = f"dados/{pasta}/frames/{frame_idx:06d}.png"
            cv2.imwrite(caminho_frame, frame_pequeno)

            # Registra dado
            dados.append({
                "frame": caminho_frame,
                "acao":  acao_para_numero(acao_atual),
                "nome":  acao_atual,
            })

            frame_idx += 1

            # Reseta ação após registrar
            acao_atual = "nada"

            time.sleep(0.25)

    finally:
        # Remove todos os hooks de teclado
        keyboard.unhook_all()

    # Salva dataset
    caminho_json = f"dados/{pasta}/dataset.json"
    with open(caminho_json, "w") as f:
        json.dump(dados, f, indent=2)

    # Resumo
    from collections import Counter
    contagem = Counter(d["nome"] for d in dados)

    print(f"\n=== Gravação finalizada! ===")
    print(f"Total de frames: {frame_idx}")
    print(f"Dataset salvo em: {caminho_json}")
    print(f"\nDistribuição de ações:")
    for nome, qtd in contagem.most_common():
        print(f"  {nome}: {qtd}")


if __name__ == "__main__":
    gravar()