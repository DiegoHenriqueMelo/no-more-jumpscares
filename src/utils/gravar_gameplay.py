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
from src.environment.fnaf_env import COORDS, ACOES, LADO_POR_ACAO, SIDE_SWITCH_DELAY, WINDOW_TITLE

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

# ─── Índice de câmera — espelha a fórmula do env (acao_num - 5) ──
CAMERA_ID = {nome: num - 5 for num, nome in ACOES.items() if nome.startswith("camera_")}

# Intervalo de captura do loop (segundos)
_LOOP_DT = 0.25

# Delay de virada de cabeça em frames de gravação (espelha SIDE_SWITCH_DELAY do env).
# Esse delay também é o único bloqueio para re-pressionamento — não há cooldown separado.
_PORTA_DELAY_FRAMES = max(1, round(SIDE_SWITCH_DELAY / _LOOP_DT))


class EstadoJogo:
    """Máquina de estado que espelha o FNAFEnv para registrar os 8 estados
    internos ao lado de cada frame gravado.

    Portas: se o personagem precisa virar a câmera para o lado da porta
    (SIDE_SWITCH_DELAY), o estado só é atualizado após _PORTA_DELAY_FRAMES
    iterações. Enquanto essa mudança está pendente, novos presses são ignorados.
    O delay em si é o único bloqueio — não há cooldown separado por steps.
    """

    def __init__(self):
        self.porta_esq:     bool  = False
        self.porta_dir:     bool  = False
        self.luz_esq:       bool  = False
        self.luz_dir:       bool  = False
        self.camera_aberta: bool  = False
        self.camera_ativa:  int   = 0       # 0 = nenhuma; 1–11 = câmera selecionada
        self.energia:       float = 100.0
        self._inicio:       float = 0.0
        self.lado_atual:    str   = "centro"
        # Cada item: [frames_restantes, "esq"|"dir", novo_valor]
        # A existência de uma pendência para um lado bloqueia novos presses daquele lado.
        self._pendencias:   list  = []

    def iniciar(self) -> None:
        self._inicio = time.perf_counter()

    def _agendar_porta(self, lado: str, novo_valor: bool, delay: int) -> None:
        if delay <= 0:
            if lado == "esq":
                self.porta_esq = novo_valor
            else:
                self.porta_dir = novo_valor
        else:
            self._pendencias.append([delay, lado, novo_valor])

    def ao_pressionar(self, nome_acao: str) -> None:
        lado_alvo = LADO_POR_ACAO.get(nome_acao)

        if nome_acao == "porta_esquerda":
            # Bloqueado apenas enquanto há uma virada de cabeça pendente para esse lado
            if any(p[1] == "esq" for p in self._pendencias):
                return
            delay = _PORTA_DELAY_FRAMES if (lado_alvo and self.lado_atual != lado_alvo) else 0
            self._agendar_porta("esq", not self.porta_esq, delay)

        elif nome_acao == "porta_direita":
            if any(p[1] == "dir" for p in self._pendencias):
                return
            delay = _PORTA_DELAY_FRAMES if (lado_alvo and self.lado_atual != lado_alvo) else 0
            self._agendar_porta("dir", not self.porta_dir, delay)

        elif nome_acao == "luz_esquerda":
            if self.luz_esq:
                self.luz_esq = False
            else:
                self.luz_esq = True
                self.luz_dir = False          # mutuamente exclusivo

        elif nome_acao == "luz_direita":
            if self.luz_dir:
                self.luz_dir = False
            else:
                self.luz_dir = True
                self.luz_esq = False          # mutuamente exclusivo

        elif nome_acao == "abrir_fechar_camera":
            self.camera_aberta = not self.camera_aberta
            if self.camera_aberta:
                self.luz_esq = False
                self.luz_dir = False
            else:
                self.camera_ativa = 0

        elif nome_acao in CAMERA_ID:
            if self.camera_aberta:
                self.camera_ativa = CAMERA_ID[nome_acao]

        # O personagem começa a virar no momento do clique
        if lado_alvo:
            self.lado_atual = lado_alvo

    def atualizar(self) -> None:
        """Chama a cada iteração do loop — processa pendências de porta e energia."""
        proximas = []
        for p in self._pendencias:
            p[0] -= 1
            if p[0] <= 0:
                if p[1] == "esq":
                    self.porta_esq = p[2]
                else:
                    self.porta_dir = p[2]
            else:
                proximas.append(p)
        self._pendencias = proximas

        itens = (
            int(self.porta_esq) + int(self.porta_dir)
            + int(self.luz_esq) + int(self.luz_dir)
            + int(self.camera_aberta)
        )
        consumo = (0.104 + min(itens, 3) * 0.100) * _LOOP_DT
        self.energia = max(0.0, self.energia - consumo)

    def tempo_ep(self) -> float:
        return time.perf_counter() - self._inicio

    def como_dict(self) -> dict:
        return {
            "porta_esq":     int(self.porta_esq),
            "porta_dir":     int(self.porta_dir),
            "luz_esq":       int(self.luz_esq),
            "luz_dir":       int(self.luz_dir),
            "camera_aberta": int(self.camera_aberta),
            "camera_ativa":  self.camera_ativa,
            "energia":       round(self.energia, 2),
            "tempo_ep":      round(min(self.tempo_ep(), 535.0), 2),
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

    estado    = EstadoJogo()
    dados     = []
    frame_idx = 0
    acao_atual = "nada"

    # Registra handlers para cada tecla
    def fazer_handler(nome_acao):
        def handler(event):
            nonlocal acao_atual
            acao_atual = nome_acao
            estado.ao_pressionar(nome_acao)
            executar_acao_no_jogo(nome_acao)
            print(f"  [{event.name}] → {nome_acao} | E:{estado.energia:.1f}% "
                  f"cam:{estado.camera_aberta} porta:{int(estado.porta_esq)}/{int(estado.porta_dir)}")
        return handler

    hooks = []
    for tecla, nome_acao in TECLAS.items():
        h = keyboard.on_press_key(tecla, fazer_handler(nome_acao))
        hooks.append(h)

    estado.iniciar()

    try:
        while not keyboard.is_pressed("f10"):
            # Captura janela do jogo (mesmo título configurado no .env)
            janelas = gw.getWindowsWithTitle(WINDOW_TITLE)
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

            # Registra dado — frame + ação + todos os 8 estados internos
            registro = {
                "frame": caminho_frame,
                "acao":  acao_para_numero(acao_atual),
                "nome":  acao_atual,
            }
            registro.update(estado.como_dict())
            dados.append(registro)

            frame_idx += 1
            acao_atual = "nada"

            estado.atualizar()
            time.sleep(_LOOP_DT)

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