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
        raise ValueError(f"Variavel obrigatoria ausente no .env: {nome}")
    try:
        return int(valor)
    except ValueError:
        raise ValueError(f"Valor invalido para {nome}: {valor}")


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
        raise ValueError(f"Variavel obrigatoria ausente no .env: {nome}")
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
STEP_DELAY = _env_float_opcional("FNAF_STEP_DELAY", 0.30)
SIDE_SWITCH_DELAY = _env_float_opcional("FNAF_SIDE_SWITCH_DELAY", 0.12)

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

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(ALTURA, LARGURA, 1),
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(ACOES))

        self.passos    = 0
        self.max_passos = 10_000
        self.energia   = 100.0
        self.porta_esq = False
        self.porta_dir = False
        self.vivo      = True
        self.lado_atual = None

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
            "passos": self.passos,
            "energia": self.energia,
            "porta_esq": self.porta_esq,
            "porta_dir": self.porta_dir,
            "morreu": False,
            "interrompido": True,
            "ocorrido": motivo,
        }

        observacao = np.zeros((ALTURA, LARGURA, 1), dtype=np.uint8)
        return observacao, 0.0, True, False, info

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
            "passos": self.passos,
            "energia": self.energia,
            "porta_esq": self.porta_esq,
            "porta_dir": self.porta_dir,
            "morreu": False,
            "interrompido": True,
            "ocorrido": motivo,
        }

        observacao = np.zeros((ALTURA, LARGURA, 1), dtype=np.uint8)
        return observacao, 0.0, True, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.passos           = 0
        self.energia          = 100.0
        self.porta_esq        = False
        self.porta_dir        = False
        self.vivo             = True
        self.lado_atual       = None
        self.contador_vitoria = 0

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
        observacao = self._capturar_observacao()
        return observacao, {}

    def step(self, acao: int):
        self.passos += 1

        if not self._janela_do_jogo_aberta():
            return self._interromper_episodio("janela do jogo nao encontrada")

        self._executar_acao(acao)
        time.sleep(STEP_DELAY)

        try:
            observacao = self._capturar_observacao()
            morreu     = self._detectar_morte()
            sobreviveu = self._detectar_vitoria()
        except Exception as erro:
            return self._interromper_episodio(f"falha ao capturar estado: {erro}")

        recompensa = self._calcular_recompensa(morreu, sobreviveu, acao)
        terminado  = morreu or sobreviveu
        truncado   = self.passos >= self.max_passos

        info = {
            "passos":    self.passos,
            "energia":   self.energia,
            "porta_esq": self.porta_esq,
            "porta_dir": self.porta_dir,
            "morreu":    morreu,
        }

        return observacao, recompensa, terminado, truncado, info

    def _executar_acao(self, acao: int):
        nome_acao = ACOES[acao]
        lado_alvo = LADO_POR_ACAO.get(nome_acao)

        if nome_acao == "nada":
            return

        if nome_acao == "porta_esquerda":
            self.porta_esq = not self.porta_esq

        if nome_acao == "porta_direita":
            self.porta_dir = not self.porta_dir

        if nome_acao in COORDS:
            x, y = COORDS[nome_acao]

            # Ao trocar de lado, move antes e espera um pouco para o jogo
            # finalizar a transicao de camera antes do clique real.
            if lado_alvo and self.lado_atual and lado_alvo != self.lado_atual:
                self.capture.mover_mouse(x, y)
                time.sleep(SIDE_SWITCH_DELAY)

            self.capture.clicar(x, y)

            if lado_alvo:
                self.lado_atual = lado_alvo

    def _calcular_recompensa(self, morreu: bool, sobreviveu: bool, acao: int) -> float:
        if morreu:
            return -100.0

        if sobreviveu:
            return +500.0

        recompensa = +1.0
        nome_acao  = ACOES[acao]

        if nome_acao in ["porta_esquerda", "porta_direita"]:
            recompensa -= 0.5

        if nome_acao in ["luz_esquerda", "luz_direita"]:
            recompensa -= 0.3

        if self.porta_esq and self.porta_dir:
            recompensa -= 2.0

        return recompensa

    def _capturar_observacao(self) -> np.ndarray:
        frame = self.capture.capturar_tela()
        frame = self.capture.redimensionar(frame, LARGURA, ALTURA)
        frame = self.capture.para_escala_cinza(frame)
        frame = np.expand_dims(frame, axis=-1)
        return frame

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
        frame = self._capturar_janela()
        resultado = cv2.matchTemplate(frame, self.template_morte, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(resultado)
        return float(max_val) > 0.70

    def _detectar_vitoria(self) -> bool:
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