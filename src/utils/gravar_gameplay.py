"""Grava gameplay humano para uso com Behavioral Cloning.

O script intercepta as teclas mapeadas e executa as ações no jogo enquanto
captura frames e registra todos os 8 estados do observation_space:

    [porta_esq, porta_dir, luz_esq, luz_dir, camera_aberta,
     camera_ativa, energia (estimada), tempo_ep]

Mapeamento de teclas
---------------------
    A   → porta_esquerda
    D   → porta_direita
    Q   → luz_esquerda
    E   → luz_direita
    Tab → abrir_fechar_camera
    1   → camera_1a      |  2 → camera_1b  |  3 → camera_1c
    4   → camera_2a      |  5 → camera_2b  |  6 → camera_3
    7   → camera_4a      |  8 → camera_4b  |  9 → camera_5
    0   → camera_6       |  - → camera_7

    F10 → para a gravação e salva o dataset

Energia estimada
-----------------
A energia é calculada pelo mesmo modelo do ambiente de treino (não lida da tela):
    consumo_por_segundo = 0.104 + itens_ativos * 0.100
    itens_ativos = min(portas + luzes + camera, 3)
Isso garante consistência com o que o agente aprende durante o treino com RL.

Uso
----
    python -m src.utils.gravar_gameplay

O dataset gerado pode ser passado diretamente para o BC:
    python -m src.agent.behavioral_cloning --dados dados/gameplay_*/dataset.json
"""
import cv2
import json
import os
import time
from collections import Counter
from datetime import datetime

import keyboard
import pyautogui
import pygetwindow as gw

from src.utils.capture import GameCapture
from src.environment.fnaf_env import COORDS, ACOES, WINDOW_TITLE

cap = GameCapture()

# ─── Configurações ────────────────────────────────────────────────────────────
FRAME_DELAY = 0.25  # segundos entre capturas (~4 fps)

# ─── Mapeamento tecla → ação ──────────────────────────────────────────────────
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

# Mapa câmera → índice numérico (igual ao FNAF_ENV: acao - 5 para cada cam)
CAMERA_PARA_IDX = {
    "camera_1a": 1,  "camera_1b": 2,  "camera_1c": 3,
    "camera_2a": 4,  "camera_2b": 5,  "camera_3":  6,
    "camera_4a": 7,  "camera_4b": 8,  "camera_5":  9,
    "camera_6":  10, "camera_7":  11,
}


def acao_para_numero(nome_acao: str) -> int:
    for numero, nome in ACOES.items():
        if nome == nome_acao:
            return numero
    return 0  # "nada" se não encontrado


def executar_acao_no_jogo(nome_acao: str):
    """Executa a ação no jogo — clique simples ou arrasto para câmera."""
    if nome_acao == "abrir_fechar_camera":
        x, y = COORDS["abrir_fechar_camera"]
        pyautogui.moveTo(x, y - 80, duration=0.05)
        pyautogui.moveTo(x, y, duration=0.15)
        time.sleep(0.08)
        pyautogui.moveTo(x, y - 80, duration=0.10)

    elif nome_acao in COORDS:
        x, y = COORDS[nome_acao]
        pyautogui.click(x, y)


class EstadoJogo:
    """Rastreia o estado interno do jogo conforme as ações do humano.

    Espelha a lógica de FNAFEnv incluindo cooldown de câmera — necessário
    para que o 9º estado (cooldown_camera > 0) seja gravado corretamente.
    """

    # Cooldown de câmera em número de frames (~4 fps × 1.0s = 4 frames)
    COOLDOWN_CAMERA_FRAMES = 4

    def __init__(self):
        self.porta_esq      = False
        self.porta_dir      = False
        self.luz_esq        = False
        self.luz_dir        = False
        self.camera_aberta  = False
        self.camera_ativa   = 0
        self.cooldown_camera = 0   # frames restantes de cooldown de câmera
        self.energia        = 100.0
        self._inicio_ep     = time.perf_counter()

    def aplicar_acao(self, nome_acao: str):
        """Atualiza o estado interno conforme a ação executada."""
        if nome_acao == "porta_esquerda":
            self.porta_esq = not self.porta_esq

        elif nome_acao == "porta_direita":
            self.porta_dir = not self.porta_dir

        elif nome_acao == "luz_esquerda":
            if self.luz_esq:
                self.luz_esq = False
            else:
                self.luz_dir = False
                self.luz_esq = True

        elif nome_acao == "luz_direita":
            if self.luz_dir:
                self.luz_dir = False
            else:
                self.luz_esq = False
                self.luz_dir = True

        elif nome_acao == "abrir_fechar_camera":
            if self.cooldown_camera == 0:
                self.camera_aberta = not self.camera_aberta
                self.cooldown_camera = self.COOLDOWN_CAMERA_FRAMES
                if not self.camera_aberta:
                    self.camera_ativa = 0
                if self.camera_aberta:
                    self.luz_esq = False
                    self.luz_dir = False

        elif nome_acao.startswith("camera_"):
            if self.camera_aberta:
                self.camera_ativa = CAMERA_PARA_IDX.get(nome_acao, 0)

    def tick(self, dt: float):
        """Avança um frame: atualiza energia e decrementa cooldowns."""
        # Energia
        itens_ativos = (
            int(self.porta_esq) + int(self.porta_dir)
            + int(self.luz_esq) + int(self.luz_dir)
            + int(self.camera_aberta)
        )
        itens_ativos = min(itens_ativos, 3)
        consumo_por_segundo = 0.104 + itens_ativos * 0.100
        self.energia = max(0.0, self.energia - consumo_por_segundo * dt)

        # Cooldown de câmera
        if self.cooldown_camera > 0:
            self.cooldown_camera -= 1

    def tempo_ep(self) -> float:
        return time.perf_counter() - self._inicio_ep

    def como_dict(self) -> dict:
        return {
            "porta_esq":      int(self.porta_esq),
            "porta_dir":      int(self.porta_dir),
            "luz_esq":        int(self.luz_esq),
            "luz_dir":        int(self.luz_dir),
            "camera_aberta":  int(self.camera_aberta),
            "camera_ativa":   self.camera_ativa,
            "energia":        round(self.energia, 2),
            "tempo_ep":       round(self.tempo_ep(), 2),
            "cooldown_camera": int(self.cooldown_camera > 0),  # 9º estado
        }


def gravar():
    print("\n" + "="*60)
    print("GRAVAÇÃO DE GAMEPLAY — Behavioral Cloning")
    print("="*60)
    print("\nMapeamento de teclas:")
    for tecla, acao in TECLAS.items():
        print(f"  [{tecla:>3}] → {acao}")
    print("\n  [F10] → Para a gravação e salva o dataset")
    print("\nOs 8 estados são rastreados automaticamente.")
    print("Energia é estimada pelo mesmo modelo do ambiente de treino.")
    print("="*60 + "\n")

    pasta  = f"gameplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(f"dados/{pasta}/frames", exist_ok=True)

    print("Você tem 5 segundos para focar o jogo...\n")
    time.sleep(5)

    dados      = []
    frame_idx  = 0
    acao_atual = "nada"
    estado     = EstadoJogo()
    t_ultimo   = time.perf_counter()

    def fazer_handler(nome_acao):
        def handler(event):
            nonlocal acao_atual
            acao_atual = nome_acao
            estado.aplicar_acao(nome_acao)
            executar_acao_no_jogo(nome_acao)
            print(f"  [{event.name:>3}] → {nome_acao} | E:{estado.energia:.1f}%")
        return handler

    hooks = []
    for tecla, nome_acao in TECLAS.items():
        h = keyboard.on_press_key(tecla, fazer_handler(nome_acao))
        hooks.append(h)

    print("Gravando... [F10 para parar]\n")

    try:
        while not keyboard.is_pressed("f10"):
            agora = time.perf_counter()
            dt    = agora - t_ultimo
            t_ultimo = agora

            # Avança um frame: energia + cooldowns
            estado.tick(dt)

            # Captura janela do jogo
            janelas = gw.getWindowsWithTitle(WINDOW_TITLE)
            if not janelas:
                print(f"Jogo não encontrado (título: '{WINDOW_TITLE}'). Aguardando...")
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

            # Registra dado com todos os 8 estados
            entrada = {
                "frame": caminho_frame,
                "acao":  acao_para_numero(acao_atual),
                "nome":  acao_atual,
            }
            entrada.update(estado.como_dict())
            dados.append(entrada)

            frame_idx += 1

            # Reseta ação após registrar (próximo frame = "nada" se nenhuma tecla)
            acao_atual = "nada"

            time.sleep(max(0, FRAME_DELAY - (time.perf_counter() - agora)))

    finally:
        keyboard.unhook_all()

    # Salva dataset
    caminho_json = f"dados/{pasta}/dataset.json"
    with open(caminho_json, "w") as f:
        json.dump(dados, f, indent=2)

    contagem = Counter(d["nome"] for d in dados)

    print(f"\n{'='*60}")
    print("Gravação finalizada!")
    print(f"  Frames gravados: {frame_idx}")
    print(f"  Dataset salvo: {caminho_json}")
    print(f"  Energia final estimada: {estado.energia:.1f}%")
    print(f"  Tempo gravado: {estado.tempo_ep():.0f}s")
    print(f"\nDistribuição de ações:")
    for nome, qtd in contagem.most_common():
        print(f"  {nome}: {qtd}")
    print(f"\nPróximo passo:")
    print(f"  python -m src.agent.behavioral_cloning --dados dados/{pasta}/dataset.json")


if __name__ == "__main__":
    gravar()
