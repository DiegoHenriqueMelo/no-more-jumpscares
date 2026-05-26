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
from src.environment.fnaf_env import (
    COORDS, ACOES, WINDOW_TITLE,
    SIDE_SWITCH_DELAY, CAMERA_EXIT_DELAY,
    CAMERA_DRAG_PIXELS, CAMERA_DRAG_DURATION,
    LADO_POR_ACAO, ACOES_CAMERA,
)

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


def executar_acao_no_jogo(nome_acao: str, estado: "EstadoJogo"):
    """Executa a ação no jogo aplicando os mesmos delays do ambiente de treino.

    - SIDE_SWITCH_DELAY: aguarda a virada de cabeça ao trocar de lado
      (esquerdo ↔ direito) antes de clicar, igual ao fnaf_env._executar_acao.
    - CAMERA_EXIT_DELAY: aguarda a animação da prancheta fechar antes de
      clicar em porta/luz, idêntico ao que o agente faz durante o treino.

    Isso elimina o duplo clique que o humano precisa dar para compensar
    o atraso do jogo — o script agora espera automaticamente, como a IA faz.
    """
    if nome_acao not in COORDS:
        return

    x, y = COORDS[nome_acao]
    lado_alvo = LADO_POR_ACAO.get(nome_acao)

    # Flags de contexto (usando ultima_acao = ação ANTERIOR ao clique atual)
    _saindo_camera = (
        estado.ultima_acao in ACOES_CAMERA
        or estado.ultima_acao == "abrir_fechar_camera"
    )
    _indo_porta_luz = nome_acao in {
        "luz_esquerda", "luz_direita", "porta_esquerda", "porta_direita"
    }
    _trocando_lado = bool(
        lado_alvo and estado.lado_atual and lado_alvo != estado.lado_atual
    )

    if nome_acao == "abrir_fechar_camera":
        # Pré-posiciona ACIMA do botão (evita hover acidental que dispara toggle)
        pyautogui.moveTo(x, y - CAMERA_DRAG_PIXELS, duration=0.05)
        if _saindo_camera or _trocando_lado:
            delay = CAMERA_EXIT_DELAY if _saindo_camera else SIDE_SWITCH_DELAY
            time.sleep(delay)
        # Arrasto para baixo (abre/fecha) + recuo para cima
        pyautogui.moveTo(x, y, duration=CAMERA_DRAG_DURATION)
        time.sleep(0.08)
        pyautogui.moveTo(x, y - CAMERA_DRAG_PIXELS, duration=CAMERA_DRAG_DURATION)

    else:
        pyautogui.moveTo(x, y, duration=0.05)
        if _saindo_camera and _indo_porta_luz:
            time.sleep(CAMERA_EXIT_DELAY)
        elif _trocando_lado:
            time.sleep(SIDE_SWITCH_DELAY)
        pyautogui.click(x, y)

    # Atualiza rastreamento de lado e última ação para o próximo clique
    if lado_alvo:
        estado.lado_atual = lado_alvo
    estado.ultima_acao = nome_acao


class EstadoJogo:
    """Rastreia o estado interno do jogo conforme as ações do humano.

    Espelha a lógica de FNAFEnv incluindo cooldowns de câmera e porta —
    necessário para bloquear ações inválidas antes de enviá-las ao jogo,
    evitando desync entre o estado interno e o que o jogo exibe.
    """

    # Cooldown de câmera em número de frames (~4 fps × 1.0s = 4 frames)
    COOLDOWN_CAMERA_FRAMES = 4
    # Cooldown de porta em número de frames (~4 fps × 0.75s ≈ 3 frames)
    # Idêntico ao fnaf_env (cooldown_porta_esq = 3).
    COOLDOWN_PORTA_FRAMES = 3

    def __init__(self):
        self.porta_esq       = False
        self.porta_dir       = False
        self.luz_esq         = False
        self.luz_dir         = False
        self.camera_aberta   = False
        self.camera_ativa    = 0
        self.cooldown_camera = 0   # frames restantes de cooldown de câmera
        self.cooldown_porta_esq = 0
        self.cooldown_porta_dir = 0
        self.energia         = 100.0
        self._inicio_ep      = time.perf_counter()
        # Rastreamento de lado/última ação — usados por executar_acao_no_jogo
        # para aplicar os mesmos delays que o fnaf_env usa durante o treino.
        self.lado_atual  = "centro"
        self.ultima_acao = "nada"

    def aplicar_acao(self, nome_acao: str) -> bool:
        """Atualiza o estado interno conforme a ação executada.

        Retorna True se a ação teve efeito, False se foi bloqueada por cooldown
        ou por estado inválido (ex: tentar porta com câmera aberta).
        O handler usa esse retorno para decidir se clica no jogo e registra
        a ação no dataset — cliques bloqueados não são gravados.
        """
        # Porta e luz não funcionam com câmera aberta
        if nome_acao in {"porta_esquerda", "porta_direita",
                         "luz_esquerda", "luz_direita"}:
            if self.camera_aberta:
                return False

        if nome_acao == "porta_esquerda":
            if self.cooldown_porta_esq > 0:
                return False
            self.porta_esq = not self.porta_esq
            self.cooldown_porta_esq = self.COOLDOWN_PORTA_FRAMES

        elif nome_acao == "porta_direita":
            if self.cooldown_porta_dir > 0:
                return False
            self.porta_dir = not self.porta_dir
            self.cooldown_porta_dir = self.COOLDOWN_PORTA_FRAMES

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
            if self.cooldown_camera > 0:
                return False
            self.camera_aberta = not self.camera_aberta
            self.cooldown_camera = self.COOLDOWN_CAMERA_FRAMES
            if not self.camera_aberta:
                self.camera_ativa = 0
            if self.camera_aberta:
                self.luz_esq = False
                self.luz_dir = False

        elif nome_acao.startswith("camera_"):
            if not self.camera_aberta:
                return False
            self.camera_ativa = CAMERA_PARA_IDX.get(nome_acao, 0)

        return True

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

        # Cooldowns
        if self.cooldown_camera > 0:
            self.cooldown_camera -= 1
        if self.cooldown_porta_esq > 0:
            self.cooldown_porta_esq -= 1
        if self.cooldown_porta_dir > 0:
            self.cooldown_porta_dir -= 1

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
            aplicou = estado.aplicar_acao(nome_acao)
            if not aplicou:
                # Ação bloqueada por cooldown ou estado inválido — não clica
                # no jogo e não registra no dataset (evita desync).
                print(f"  [{event.name:>3}] → {nome_acao} BLOQUEADO (cooldown/câmera)")
                return
            acao_atual = nome_acao
            executar_acao_no_jogo(nome_acao, estado)
            print(f"  [{event.name:>3}] → {nome_acao} | lado:{estado.lado_atual} | E:{estado.energia:.1f}%")
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
