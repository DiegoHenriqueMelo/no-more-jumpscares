import gymnasium as gym
import numpy as np
import cv2
import time
import os
import subprocess
import unicodedata
from pathlib import Path
from gymnasium import spaces
from src.utils.capture import GameCapture

def _carregar_env(caminho: str = ".env") -> None:
    if not os.path.exists(caminho):
        return

    with open(caminho, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            conteudo = linha.strip()
            if not conteudo or conteudo.startswith("#") or "=" not in conteudo:
                continue

            chave, valor = conteudo.split("=", 1)
            chave = chave.strip()
            valor = valor.strip().strip('"').strip("'")
            os.environ.setdefault(chave, valor)


_carregar_env()

LARGURA = 84
ALTURA  = 84

ACOES = {
    0:  "nada",
    1:  "porta_esquerda",
    2:  "porta_direita",
    3:  "luz_esquerda",
    4:  "luz_direita",
    5:  "abrir_fechar_camera",
    6:  "camera_1a",
    7:  "camera_1b",
    8:  "camera_1c",
    9:  "camera_2a",
    10: "camera_2b",
    11: "camera_3",
    12: "camera_4a",
    13: "camera_4b",
    14: "camera_5",
    15: "camera_6",
    16: "camera_7",
}

ACOES_CAMERA = {acao for acao in ACOES.values() if acao.startswith("camera_")}
ACOES_LADO_ESQUERDO = {"porta_esquerda", "luz_esquerda"}
ACOES_LADO_DIREITO = {"porta_direita", "luz_direita", "abrir_fechar_camera"} | ACOES_CAMERA

LADO_POR_ACAO = {acao: "esquerdo" for acao in ACOES_LADO_ESQUERDO}
LADO_POR_ACAO.update({acao: "direito" for acao in ACOES_LADO_DIREITO})

def _env_int_obrigatorio(nome: str) -> int:
    valor = os.getenv(nome)
    if valor is None or valor.strip() == "":
        return 0
    try:
        return int(valor)
    except ValueError:
        return 0


def _env_int_opcional(nome: str, padrao: int) -> int:
    valor = os.getenv(nome)
    if valor is None or valor.strip() == "":
        return padrao
    try:
        return int(valor)
    except ValueError:
        return padrao


def _env_str_obrigatorio(nome: str) -> str:
    valor = os.getenv(nome)
    if valor is None or valor.strip() == "":
        return ""
    return valor.strip()


def _env_float_opcional(nome: str, padrao: float) -> float:
    valor = os.getenv(nome)
    if valor is None or valor.strip() == "":
        return padrao

    try:
        convertido = float(valor)
    except ValueError:
        raise ValueError(f"Valor invalido para {nome}: {valor}")

    if convertido < 0:
        raise ValueError(f"Valor invalido para {nome}: {valor}. Use numero >= 0")

    return convertido

def _env_str_opcional(nome: str, padrao: str = "") -> str:
    valor = os.getenv(nome)
    if valor is None:
        return padrao
    return valor.strip() or padrao



def _env_coord(acao: str) -> tuple[int, int]:
    prefixo = f"FNAF_COORD_{acao.upper()}".replace("-", "_")
    x = _env_int_obrigatorio(f"{prefixo}_X")
    y = _env_int_obrigatorio(f"{prefixo}_Y")
    return x, y


WINDOW_TITLE = _env_str_obrigatorio("FNAF_WINDOW_TITLE")
GAME_EXECUTABLE_PATH = _env_str_opcional("FNAF_EXECUTABLE_PATH", "")
REABRIR_ESPERA_SEGUNDOS = max(1, _env_int_opcional("FNAF_REABRIR_ESPERA_SEGUNDOS", 15))
POS_ALT_ENTER_ESPERA_SEGUNDOS = max(1, _env_int_opcional("FNAF_POS_ALT_ENTER_ESPERA_SEGUNDOS", 3))
RESET_CLICK = (
    _env_int_obrigatorio("FNAF_RESET_CLICK_X"),
    _env_int_obrigatorio("FNAF_RESET_CLICK_Y"),
)
STEP_DELAY = _env_float_opcional("FNAF_STEP_DELAY", 0.25)
SIDE_SWITCH_DELAY = _env_float_opcional("FNAF_SIDE_SWITCH_DELAY", 0.12)
CAMERA_EXIT_DELAY = _env_float_opcional("FNAF_CAMERA_EXIT_DELAY", 0.5)
CAMERA_DRAG_PIXELS   = _env_int_opcional("FNAF_CAMERA_DRAG_PIXELS", 80)
CAMERA_DRAG_DURATION = _env_float_opcional("FNAF_CAMERA_DRAG_DURATION", 0.15)

COORDS = {
    acao: _env_coord(acao)
    for acao in ACOES.values()
    if acao != "nada"
}


class FNAFEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.capture          = GameCapture()
        self.render_mode      = render_mode
        self.contador_vitoria = 0
        self._carregar_templates()

        self.observation_space = spaces.Dict({
            "imagem": spaces.Box(
                low=0, high=255,
                shape=(ALTURA, LARGURA, 1),
                dtype=np.uint8
            ),
            "estados": spaces.Box(
                low=0, high=1,
                shape=(7,),
                dtype=np.float32
            )
        })
        self.action_space = spaces.Discrete(len(ACOES))

        self.passos    = 0
        self.max_passos = 10_000
        self.energia   = 100.0
        self.tempo_jogo = 0.0
        self.luz_esq = False
        self.luz_dir = False
        self.luz_esq_timer = 0
        self.luz_dir_timer = 0
        self.porta_esq = False
        self.porta_dir = False
        self.camera_aberta = False
        self.camera_ativa = 0
        self.vivo      = True
        self.lado_atual = None
        self.ultima_acao = None
        self.penultima_acao = None
        self.ultimo_update_energia = None
        self.passos_sem_camera     = 0

    def _janela_do_jogo_aberta(self) -> bool:
        import pygetwindow as gw
        return bool(gw.getWindowsWithTitle(WINDOW_TITLE))

    @staticmethod
    def _normalizar_texto(texto: str) -> str:
        texto = unicodedata.normalize("NFKD", texto)
        texto = "".join(char for char in texto if not unicodedata.combining(char))
        return texto.lower()

    @staticmethod
    def _caminhos_desktop() -> list[Path]:
        candidatos = [
            Path.home() / "Desktop",
            Path(os.getenv("USERPROFILE", "")) / "Desktop",
            Path(os.getenv("PUBLIC", "")) / "Desktop",
        ]

        one_drive = os.getenv("OneDrive")
        if one_drive:
            candidatos.append(Path(one_drive) / "Desktop")

        unicos: list[Path] = []
        vistos: set[str] = set()
        for caminho in candidatos:
            chave = str(caminho).strip().lower()
            if not chave or chave in vistos:
                continue
            vistos.add(chave)
            unicos.append(caminho)
        return unicos

    def _descobrir_atalho_desktop(self) -> Path | None:
        palavras_chave = ("five nights", "freddy", "fnaf")
        extensoes_validas = {".lnk", ".url", ".exe"}

        for desktop in self._caminhos_desktop():
            if not desktop.exists() or not desktop.is_dir():
                continue

            arquivos = sorted(
                [arquivo for arquivo in desktop.iterdir() if arquivo.is_file()],
                key=lambda item: item.name.lower(),
            )

            for arquivo in arquivos:
                if arquivo.suffix.lower() not in extensoes_validas:
                    continue

                nome_normalizado = self._normalizar_texto(arquivo.stem)
                if any(chave in nome_normalizado for chave in palavras_chave):
                    return arquivo

        return None

    def _resolver_caminho_jogo(self) -> Path | None:
        if GAME_EXECUTABLE_PATH:
            texto_expandido = os.path.expandvars(os.path.expanduser(GAME_EXECUTABLE_PATH))
            caminho = Path(texto_expandido)
            if caminho.exists() and caminho.is_file():
                return caminho

            nome_apenas = Path(GAME_EXECUTABLE_PATH).name
            for desktop in self._caminhos_desktop():
                candidato = desktop / nome_apenas
                if candidato.exists() and candidato.is_file():
                    return candidato

            print(
                "[FALLBACK] Caminho invalido em FNAF_EXECUTABLE_PATH: "
                f"{GAME_EXECUTABLE_PATH}"
            )

        encontrado = self._descobrir_atalho_desktop()
        if encontrado is not None:
            print(f"[FALLBACK] Usando atalho detectado na area de trabalho: {encontrado}")
        return encontrado

    @staticmethod
    def _abrir_arquivo(path_arquivo: Path) -> bool:
        try:
            if os.name == "nt":
                os.startfile(str(path_arquivo))
            else:
                subprocess.Popen([str(path_arquivo)], cwd=str(path_arquivo.parent))
            return True
        except Exception:
            return False

    def _abrir_jogo_fallback(self) -> bool:
        caminho = self._resolver_caminho_jogo()
        if caminho is None:
            print(
                "[FALLBACK] Nao foi encontrado executavel/atalho do jogo. "
                "Configure FNAF_EXECUTABLE_PATH com .exe ou .lnk."
            )
            return False

        if not self._abrir_arquivo(caminho):
            print(f"[FALLBACK] Falha ao abrir jogo automaticamente: {caminho}")
            return False

        print("[FALLBACK] Jogo fechado detectado. Relancando executavel...")
        time.sleep(REABRIR_ESPERA_SEGUNDOS)

        if not self.capture.focar_janela(WINDOW_TITLE):
            print("[FALLBACK] Janela nao encontrada apos relancamento.")
            return False

        self.capture.atalho("alt", "enter")
        time.sleep(POS_ALT_ENTER_ESPERA_SEGUNDOS)
        self.capture.focar_janela(WINDOW_TITLE)
        print("[FALLBACK] Jogo recolocado em modo janela (ALT+ENTER).")
        return True

    def _interromper_episodio(self, motivo: str):
        recuperado = self._abrir_jogo_fallback()
        if not recuperado:
            motivo = f"{motivo} (fallback sem sucesso)"

        info = {
            "passos":       self.passos,
            "energia":      self.energia,
            "tempo_real":   time.perf_counter() - (self.episode_start_time or time.perf_counter()),
            "porta_esq":    self.porta_esq,
            "porta_dir":    self.porta_dir,
            "camera_aberta": self.camera_aberta,
            "camera_ativa": self.camera_ativa,
            "morreu":       False,
            "interrompido": True,
            "ocorrido":     motivo,
        }

        observacao = {
            "imagem": np.zeros((ALTURA, LARGURA, 1), dtype=np.uint8),
            "estados": np.zeros(7, dtype=np.float32)
        }
        return observacao, 0.0, True, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not WINDOW_TITLE:
            raise RuntimeError(
                "FNAF_WINDOW_TITLE nao configurado no .env. "
                "Configure as variaveis obrigatorias antes de executar."
            )

        self.passos           = 0
        self.energia          = 100.0
        self.tempo_jogo       = 0.0
        self.luz_esq          = False
        self.luz_dir          = False
        self.luz_esq_timer    = 0
        self.luz_dir_timer    = 0
        self.porta_esq        = False
        self.porta_dir        = False
        self.camera_aberta    = False
        self.camera_ativa     = 0
        self.vivo             = True
        self.lado_atual       = None
        self.ultima_acao      = None
        self.penultima_acao   = None
        self.contador_vitoria  = 0
        self.ultimo_update_energia = None
        self.episode_start_time    = None
        self.passos_sem_camera     = 0

        if not self._janela_do_jogo_aberta():
            self._abrir_jogo_fallback()

        if not self.capture.focar_janela(WINDOW_TITLE):
            raise RuntimeError(
                "Janela do jogo nao encontrada. "
                "Configure FNAF_EXECUTABLE_PATH no .env para fallback automatico."
            )
        time.sleep(0.5)

        self.capture.clicar(*RESET_CLICK)
        time.sleep(15)
        self.capture.clicar(*RESET_CLICK)
        time.sleep(20)

        print("Reset completo — noite iniciada!")
        agora = time.perf_counter()
        self.ultimo_update_energia = agora
        self.episode_start_time    = agora
        observacao = self._capturar_observacao()
        return observacao, {}

    def step(self, acao: int):
        self.passos += 1

        if not self._janela_do_jogo_aberta():
            return self._interromper_episodio("janela do jogo nao encontrada")

        # Quando energia acabou, desliga tudo mas não encerra ainda —
        # no FNAF1 o Freddy demora alguns segundos para aparecer após a
        # energia zerar. Encerrar aqui causaria reset() durante a animação
        # de morte, corrompendo o estado do jogo. O episódio termina quando
        # _detectar_morte() confirmar via template, igual ao caminho normal.
        if self.energia <= 0:
            self.porta_esq = False
            self.porta_dir = False
            self.luz_esq = False
            self.luz_dir = False
            self.camera_aberta = False

        acao_valida = self._executar_acao(acao)
        time.sleep(STEP_DELAY)
        
        self._atualizar_luzes()
        self._atualizar_energia()
        self._atualizar_tempo()

        if self.camera_aberta:
            self.passos_sem_camera = 0
        else:
            self.passos_sem_camera += 1

        try:
            observacao = self._capturar_observacao()
            morreu     = self._detectar_morte()
            sobreviveu = self._detectar_vitoria()
        except Exception as erro:
            return self._interromper_episodio(f"falha ao capturar estado: {erro}")

        recompensa = self._calcular_recompensa(morreu, sobreviveu, acao, acao_valida)
        terminado  = morreu or sobreviveu
        truncado   = self.passos >= self.max_passos

        info = {
            "passos":         self.passos,
            "energia":        self.energia,
            "tempo":          self.tempo_jogo,
            "tempo_real":     time.perf_counter() - (self.episode_start_time or time.perf_counter()),
            "luz_esq":        self.luz_esq,
            "luz_dir":        self.luz_dir,
            "porta_esq":      self.porta_esq,
            "porta_dir":      self.porta_dir,
            "camera_aberta":  self.camera_aberta,
            "camera_ativa":   self.camera_ativa,
            "morreu":         morreu,
            "acao_valida":    acao_valida,
        }

        return observacao, recompensa, terminado, truncado, info

    def _executar_acao(self, acao: int) -> bool:
        """Executa ação e retorna True se teve efeito, False se foi inválida."""
        nome_acao = ACOES[acao]
        lado_alvo = LADO_POR_ACAO.get(nome_acao)

        if nome_acao == "nada":
            self.penultima_acao = self.ultima_acao
            self.ultima_acao = nome_acao
            return True

        # Ações de porta/luz só funcionam quando NÃO está na câmera
        if nome_acao in ["porta_esquerda", "porta_direita", "luz_esquerda", "luz_direita"]:
            if self.camera_aberta:
                return False  # Ação inválida - está na câmera
            
            if nome_acao == "porta_esquerda":
                self.porta_esq = not self.porta_esq
            elif nome_acao == "porta_direita":
                self.porta_dir = not self.porta_dir
            elif nome_acao == "luz_esquerda":
                self.luz_esq = True
                self.luz_esq_timer = 1  # Desliga após 1 step
            elif nome_acao == "luz_direita":
                self.luz_dir = True
                self.luz_dir_timer = 1  # Desliga após 1 step

        # Abrir/fechar câmera sempre funciona
        elif nome_acao == "abrir_fechar_camera":
            self.camera_aberta = not self.camera_aberta
            if not self.camera_aberta:
                self.camera_ativa = 0
            # Ao abrir câmera, desliga luzes (não fazem sentido no contexto)
            if self.camera_aberta:
                self.luz_esq = False
                self.luz_dir = False
                self.luz_esq_timer = 0
                self.luz_dir_timer = 0
            # Arrasta o mouse de cima para baixo até o botão para acionar
            # a animação de arrastar a prancheta do jogo
            if nome_acao in COORDS:
                x, y = COORDS[nome_acao]
                self.capture.mover_mouse(x, y - CAMERA_DRAG_PIXELS)
                self.capture.arrastar_para(x, y, duration=CAMERA_DRAG_DURATION)

        # Trocar de câmera só funciona se câmera estiver aberta
        elif nome_acao.startswith("camera_"):
            if not self.camera_aberta:
                return False  # Ação inválida - câmera fechada
            self.camera_ativa = acao - 5

        if nome_acao in COORDS:
            x, y = COORDS[nome_acao]

            # Ao trocar de lado, move antes e espera um pouco para o jogo
            # finalizar a transicao de camera antes do clique real.
            if lado_alvo and self.lado_atual and lado_alvo != self.lado_atual:
                self.capture.mover_mouse(x, y)
                time.sleep(SIDE_SWITCH_DELAY)
            # Ao sair das cameras para luz/porta direita, espera para a camera acompanhar
            elif self.ultima_acao in ACOES_CAMERA and nome_acao in {"luz_direita", "porta_direita"}:
                self.capture.mover_mouse(x, y)
                time.sleep(CAMERA_EXIT_DELAY)

            self.capture.clicar(x, y)

            if lado_alvo:
                self.lado_atual = lado_alvo

        self.penultima_acao = self.ultima_acao
        self.ultima_acao = nome_acao
        return True

    def _atualizar_luzes(self):
        """Luzes desligam automaticamente após 1 step (comportamento realista)."""
        if self.luz_esq_timer > 0:
            self.luz_esq_timer -= 1
            if self.luz_esq_timer == 0:
                self.luz_esq = False
        
        if self.luz_dir_timer > 0:
            self.luz_dir_timer -= 1
            if self.luz_dir_timer == 0:
                self.luz_dir = False
    
    def _atualizar_energia(self):
        agora = time.perf_counter()
        if self.ultimo_update_energia is None:
            self.ultimo_update_energia = agora
            return
        
        dt = agora - self.ultimo_update_energia
        
        usage = 1
        usage += int(self.porta_esq)
        usage += int(self.porta_dir)
        usage += int(self.luz_esq)
        usage += int(self.luz_dir)
        usage += int(self.camera_aberta)
        usage = min(usage, 4)
        
        consumo_por_segundo = usage * 0.1
        self.energia -= consumo_por_segundo * dt
        self.energia = max(0.0, self.energia)
        
        self.ultimo_update_energia = agora
    
    def _atualizar_tempo(self):
        self.tempo_jogo += STEP_DELAY
    
    def _energia_esperada(self) -> float:
        """Energia esperada (%) com base no progresso da noite, seguindo os thresholds do jogo."""
        checkpoints = [
            (0,   100.0),
            (89,   85.0),   # 1AM
            (178,  60.0),   # 2AM
            (267,  40.0),   # 3AM
            (356,  25.0),   # 4AM
            (445,  15.0),   # 5AM
            (535,   5.0),   # 6AM
        ]
        t = self.tempo_jogo
        for i in range(len(checkpoints) - 1):
            t0, e0 = checkpoints[i]
            t1, e1 = checkpoints[i + 1]
            if t <= t1:
                frac = (t - t0) / (t1 - t0)
                return e0 + frac * (e1 - e0)
        return 5.0

    def _calcular_recompensa(self, morreu: bool, sobreviveu: bool, acao: int, acao_valida: bool) -> float:
        if self.energia <= 0:
            return -500.0

        if morreu:
            return -500.0

        if sobreviveu:
            return +1000.0

        if not acao_valida:
            return 0.0

        # "nada" e demais ações partem de recompensa neutra
        recompensa = 0.0

        # Bônus por progresso no tempo (incentiva sobreviver mais)
        progresso = self.tempo_jogo / 535.0
        recompensa += progresso * 0.5  # até +0.5 no final

        nome_acao = ACOES[acao]

        # Penalidade por ação repetida — inclui "nada" para evitar passividade
        if nome_acao == self.ultima_acao:
            if nome_acao in ["porta_esquerda", "porta_direita", "luz_esquerda", "luz_direita"]:
                if nome_acao == self.penultima_acao:
                    recompensa -= 1.5  # 3x seguidas = spam
            else:
                recompensa -= 1.0

        # Pequena penalidade por usar luzes (gasta energia sem observar)
        if nome_acao in ["luz_esquerda", "luz_direita"]:
            recompensa -= 0.2

        # Penalidade por ter ambas as portas fechadas (raramente necessário)
        if self.porta_esq and self.porta_dir:
            recompensa -= 1.0

        # Penalidade por inatividade da câmera — Foxy corre a cada ~5s sem câmera
        if self.passos_sem_camera > 20:
            excesso = self.passos_sem_camera - 20
            recompensa -= min(excesso * 0.05, 1.0)

        # Penalidade por energia abaixo do esperado para o momento da noite
        deficit = max(0.0, self._energia_esperada() - self.energia)
        recompensa -= deficit * 0.02

        recompensa = max(recompensa, -2.0)

        return recompensa

    def _capturar_observacao(self) -> dict:
        frame = self.capture.capturar_tela()
        frame = self.capture.redimensionar(frame, LARGURA, ALTURA)
        frame = self.capture.para_escala_cinza(frame)
        frame = np.expand_dims(frame, axis=-1)

        estados = np.array([
            float(self.porta_esq),
            float(self.porta_dir),
            float(self.luz_esq),
            float(self.luz_dir),
            float(self.camera_aberta),
            float(self.camera_ativa) / 11.0,
            float(self.energia) / 100.0
        ], dtype=np.float32)

        return {"imagem": frame, "estados": estados}

    def _carregar_templates(self):
        refs = Path(__file__).parent.parent / "utils" / "referencias"

        def _ler_primeira_existente(*nomes):
            for nome in nomes:
                caminho = refs / nome
                if caminho.exists():
                    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
                    if imagem is not None:
                        return imagem, nome
            return None, None

        morte_img, morte_nome = _ler_primeira_existente("morte.png", "morte.jpg", "morte.jpeg")
        vitoria_img, vitoria_nome = _ler_primeira_existente("vitoria.png", "vitoria.jpg", "vitoria.jpeg")

        faltando = []
        if morte_img is None:
            faltando.append("morte.(png/jpg)")
        if vitoria_img is None:
            faltando.append("vitoria.(png/jpg)")

        if faltando:
            raise FileNotFoundError(
                "Imagens de referência não encontradas em src/utils/referencias/: "
                + ", ".join(faltando)
                + ". Rode: python -m src.utils.calibrar morte e python -m src.utils.calibrar vitoria"
            )

        print(f"Referências carregadas: {morte_nome}, {vitoria_nome}")

        # Resolução das referências — o frame capturado será redimensionado para isso
        self._ref_size = (morte_img.shape[1], morte_img.shape[0])  # (w, h) = (1280, 720)

        # Recorta só o texto "Game Over" (canto inferior direito de morte.jpg)
        h, w = morte_img.shape
        self.template_morte = morte_img[int(h * 0.88):, int(w * 0.82):]

        # Recorta só o texto "6 AM" (centro de vitoria.png)
        h, w = vitoria_img.shape
        self.template_vitoria = vitoria_img[int(h * 0.38):int(h * 0.58), int(w * 0.38):int(w * 0.62)]

    def _capturar_janela(self) -> np.ndarray:
        """Captura apenas a janela do jogo e redimensiona para a resolução de referência."""
        import pygetwindow as gw
        janelas = gw.getWindowsWithTitle(WINDOW_TITLE)
        if not janelas:
            raise RuntimeError("janela do jogo nao encontrada")

        win = janelas[0]
        regiao = {
            "left":   win.left,
            "top":    win.top,
            "width":  win.width,
            "height": win.height,
        }
        frame = self.capture.capturar_tela(regiao)

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(cinza, self._ref_size)

    def _detectar_morte(self) -> bool:
        # Ignora detecção nos primeiros 120 passos (~30s de episódio / ~60s de jogo
        # real contando o reset) para evitar detectar a tela de Game Over do episódio
        # anterior, que pode persistir durante a transição de reset.
        if self.passos < 120:
            return False
        
        frame = self._capturar_janela()
        resultado = cv2.matchTemplate(frame, self.template_morte, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(resultado)
        return float(max_val) > 0.70

    def _detectar_vitoria(self) -> bool:
        # Ignora detecção nos primeiros 30 passos (~7.5s) para o jogo terminar de carregar
        if self.passos < 30:
            return False
        
        frame = self._capturar_janela()
        resultado = cv2.matchTemplate(frame, self.template_vitoria, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(resultado)

        if float(max_val) > 0.70:
            self.contador_vitoria += 1
        else:
            self.contador_vitoria = 0

        return self.contador_vitoria >= 3

    def render(self):
        pass

    def close(self):
        pass