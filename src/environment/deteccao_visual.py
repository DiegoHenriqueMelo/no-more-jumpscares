"""
Detecção visual de ocupação (animatrônicos) por desvio do "vazio".

Filosofia (ver docs/VISAO_COMPUTACIONAL_OBSERVABILIDADE.md):

- NÃO faz reconhecimento facial. Detecta se uma região que *deveria* estar
  vazia tem "algo" ali — robusto a pose, base e distorção do rosto, porque
  detecta "tem algo que não deveria estar", não um rosto específico.

- Saída por região: (peso_perigo, confianca), ambos em [0, 1].
    * peso_perigo: quão diferente do vazio a região está agora (maior = mais
      provável que haja um animatrônico).
    * confianca: quão confiável é a leitura neste frame. Estática, glitch ou
      tela preta derrubam a confiança em vez de cuspir um falso negativo.

- Tolerante a translação: compara via matchTemplate da referência do vazio
  (recortada com uma margem) contra a ROI atual. Assim o balanço involuntário
  da câmera desliza o template e NÃO dispara falso positivo.

- Fallback gracioso: região sem referência carregada → (0.0, 0.0). Nada quebra,
  igual ao template de câmera do fnaf_env quando ausente.

Este módulo é PURO sensor: não importa nada do RL e não tem efeito colateral.
As Etapas 1 e 2 do plano servem para calibrar as ROIs e os limiares aqui com o
jogo aberto, antes de qualquer integração ao espaço de observação (Etapa 3).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# ─── Diretório das referências do "vazio" ────────────────────────────────────
REFS_VAZIO = Path(__file__).parent.parent / "utils" / "referencias" / "vazio"

# ─── Tamanho canônico de comparação ──────────────────────────────────────────
# ROI atual e referência são redimensionadas para isto antes de comparar, então
# a detecção independe da resolução da janela do jogo.
TAM_CANONICO = (200, 200)  # (w, h)

# Margem recortada da referência para virar "template" deslizável dentro da ROI.
# 0.12 → o template desliza ±24px no canônico, cobrindo o balanço lento da câmera.
MARGEM_DESLIZE = 0.12

# Diferença de pixel (0-255, após blur + CLAHE) acima da qual consideramos que
# aquele ponto "mudou" em relação ao vazio. O perigo é a FRAÇÃO de pontos que
# mudaram — sensível a um intruso localizado mesmo numa ROI com muita UI fixa.
# Com o contraste normalizado (CLAHE), o vazio fica ~0 e o vulto sobe bem.
LIMIAR_DIFF = 25.0

# ─── Limiares de confiança (AJUSTAR na Etapa 1/2 com o diagnóstico) ───────────
LIMIAR_FLAT         = 0.5     # desvio-padrão abaixo disso → quadro CHAPADO (preto/transição, sem info)
                              # baixo de propósito: só pega quadro REALMENTE chapado; cena escura com
                              # grão (std normal do FNAF) passa e é tratada como confiável.
CONF_PRETO          = 0.15    # confiança quando o quadro está chapado (sem informação)
LIMIAR_ESTATICA     = 900.0   # variância do Laplaciano acima disso → estática/glitch
CONF_ESTATICA       = 0.25    # confiança quando há estática forte

# Abaixo desta confiança a leitura de perigo é considerada não confiável
# ("não sei agora") — usado no diagnóstico e, na Etapa 3, pelo env.
CONF_GATE           = 0.5


# ─── Configuração das regiões ────────────────────────────────────────────────
# ROI em FRAÇÕES da janela (x0, y0, x1, y1), 0..1 — independe da resolução.
# camera_ativa espelha a fórmula do fnaf_env (numero_da_acao - 5); None = porta,
# que é vista no escritório (câmera fechada), não numa tab.
#
# ATENÇÃO: estas frações são PALPITES iniciais. A Etapa 1/2 existe justamente
# para afiná-las olhando o diagnóstico ao vivo. Não confie nelas sem validar.
REGIOES: dict[str, dict] = {
    # ── Lado ESQUERDO ────────────────────────────────────────────────────────
    # O animatrônico aparece no VÃO DA PORTA; a SOMBRA dele aparece na JANELA ao
    # lado. Dois pontos de observação do MESMO lado (a sombra costuma avisar mais
    # cedo). Ficam separados de propósito: ROI menor = a sombra ocupa fração maior
    # do recorte = sinal mais forte. A Etapa 3 decide se agrega num único
    # "perigo esquerdo" ligado à ação de fechar a porta esquerda (vínculo
    # APRENDIDO pela IA, não cabeado aqui).
    "porta_esq":  {"roi": (0.00, 0.20, 0.17, 0.95), "camera_ativa": None, "lado": "esq"},
    "janela_esq": {"roi": (0.15, 0.28, 0.30, 0.82), "camera_ativa": None, "lado": "esq"},

    # ── Lado DIREITO ─────────────────────────────────────────────────────────
    # O animatrônico aparece na JANELA. A ação continua sendo fechar a porta.
    "janela_dir": {"roi": (0.83, 0.20, 1.00, 0.95), "camera_ativa": None, "lado": "dir"},

    # ── Câmeras (a sala ocupa quase toda a tela quando a tab está aberta) ─────
    "cam_1c": {"roi": (0.08, 0.10, 0.92, 0.88), "camera_ativa": 3, "lado": None},
    "cam_2a": {"roi": (0.08, 0.10, 0.92, 0.88), "camera_ativa": 4, "lado": None},
    "cam_2b": {"roi": (0.08, 0.10, 0.92, 0.88), "camera_ativa": 5, "lado": None},
    "cam_4a": {"roi": (0.08, 0.10, 0.92, 0.88), "camera_ativa": 7, "lado": None},
    "cam_4b": {"roi": (0.08, 0.10, 0.92, 0.88), "camera_ativa": 8, "lado": None},
}

CAMERAS = ("cam_1c", "cam_2a", "cam_2b", "cam_4a", "cam_4b")

# Regiões agrupadas por lado da porta — a Etapa 3 usa para montar UM sinal de
# "perigo" por lado (ex.: max das ROIs do lado) ligado à ação de fechar a porta.
REGIOES_POR_LADO = {
    "esq": [nome for nome, cfg in REGIOES.items() if cfg.get("lado") == "esq"],
    "dir": [nome for nome, cfg in REGIOES.items() if cfg.get("lado") == "dir"],
}

# camera_ativa → nome da região (para a Etapa 3 saber qual câmera está aberta)
CAMERA_ATIVA_PARA_REGIAO = {
    cfg["camera_ativa"]: nome
    for nome, cfg in REGIOES.items()
    if cfg["camera_ativa"] is not None
}

# Slots de perigo expostos ao agente no vetor de observação (ordem FIXA).
# 'esq'/'dir' agregam (max) as ROIs daquele lado; cada câmera-alvo é um slot.
# O nome do slot de câmera é igual ao nome da região (cam_1c, ...).
SLOTS_PERIGO = ("esq", "dir") + CAMERAS

# ─── Overrides de ROI (ajuste fino sem editar código) ────────────────────────
# `calibrar roi <regiao>` grava aqui as frações desenhadas com o mouse na tela
# do jogo. O detector carrega na inicialização e usa por cima das frações-padrão
# de REGIOES. Assim dá para reposicionar uma ROI mal enquadrada sem mexer no
# código — e o ajuste persiste.
ARQ_OVERRIDES_ROI = REFS_VAZIO / "rois.json"


def _carregar_overrides_roi() -> dict:
    if ARQ_OVERRIDES_ROI.exists():
        try:
            return json.loads(ARQ_OVERRIDES_ROI.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


_OVERRIDES_ROI = _carregar_overrides_roi()


def roi_da_regiao(regiao: str) -> tuple:
    """ROI efetiva da região: o override (se houver) por cima do padrão."""
    ov = _OVERRIDES_ROI.get(regiao)
    if ov and len(ov) == 4:
        return tuple(ov)
    return REGIOES[regiao]["roi"]


def salvar_override_roi(regiao: str, roi) -> Path:
    """Grava/atualiza a ROI da região no override JSON."""
    ov = _carregar_overrides_roi()
    ov[regiao] = [round(float(v), 4) for v in roi]
    ARQ_OVERRIDES_ROI.parent.mkdir(parents=True, exist_ok=True)
    ARQ_OVERRIDES_ROI.write_text(
        json.dumps(ov, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _OVERRIDES_ROI[regiao] = ov[regiao]
    return ARQ_OVERRIDES_ROI


def recortar_roi(frame_cinza: np.ndarray, regiao: str) -> np.ndarray:
    """Recorta a ROI da região e normaliza para o tamanho canônico (cinza)."""
    x0f, y0f, x1f, y1f = roi_da_regiao(regiao)
    h, w = frame_cinza.shape[:2]
    x0, x1 = int(x0f * w), int(x1f * w)
    y0, y1 = int(y0f * h), int(y1f * h)
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    recorte = frame_cinza[y0:y1, x0:x1]
    if recorte.size == 0:
        return np.zeros(TAM_CANONICO[::-1], dtype=np.uint8)
    return cv2.resize(recorte, TAM_CANONICO)


def _para_cinza(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


# CLAHE: equaliza o contraste local. As cenas do FNAF são muito escuras — o vulto
# difere pouco do fundo em brilho ABSOLUTO. Normalizar o contraste (aplicado igual
# na ROI e na referência) torna a diferença estrutural visível, sem inventar
# diferença onde a cena é idêntica.
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _preparar(img: np.ndarray) -> np.ndarray:
    """Suaviza (tira grão) + normaliza contraste (CLAHE). Base da comparação."""
    return _CLAHE.apply(cv2.GaussianBlur(img, (5, 5), 0))


class DetectorAnimatronicos:
    """Sensor de ocupação por região. Carrega as referências do vazio uma vez
    e responde (peso_perigo, confianca) por região a cada frame."""

    def __init__(self, refs_dir: Path | str = REFS_VAZIO):
        self.refs_dir = Path(refs_dir)
        self._refs: dict[str, list[np.ndarray]] = {}
        self.carregar_referencias()

    # ── carregamento ─────────────────────────────────────────────────────────
    def carregar_referencias(self) -> None:
        """(Re)carrega referencias/vazio/{regiao}*.png. Cada região pode ter
        VÁRIAS variantes do vazio (ex.: porta_esq.png + porta_esq_fechada.png,
        ou os dois extremos do balanço de uma câmera) — o perigo é medido contra
        a variante mais próxima. Região sem nenhuma variante fica sem referência
        → detecção devolve (0.0, 0.0) (fallback neutro)."""
        self._refs.clear()
        for regiao in REGIOES:
            variantes = []
            for caminho in sorted(self.refs_dir.glob(f"{regiao}*.png")):
                img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    variantes.append(cv2.resize(img, TAM_CANONICO))
            if variantes:
                self._refs[regiao] = variantes

    def regioes_calibradas(self) -> list[str]:
        return sorted(self._refs.keys())

    def regioes_faltando(self) -> list[str]:
        return [r for r in REGIOES if r not in self._refs]

    # ── detecção ─────────────────────────────────────────────────────────────
    def detectar_regiao(self, frame: np.ndarray, regiao: str) -> tuple[float, float]:
        """Retorna (peso_perigo, confianca) ∈ [0,1] para uma região.

        Sem referência carregada → (0.0, 0.0): o agente aprende a tratar
        confianca=0 como "sem informação", em vez de levar um falso negativo."""
        refs = self._refs.get(regiao)
        if not refs:
            return 0.0, 0.0

        roi = recortar_roi(_para_cinza(frame), regiao)
        confianca = self._confianca(roi)
        perigo = self._perigo(roi, refs)
        return perigo, confianca

    def detectar_todas(self, frame: np.ndarray) -> dict[str, tuple[float, float]]:
        """Roda a detecção em TODAS as regiões. Útil só para o diagnóstico —
        no jogo real cada câmera só é observável quando a tab dela está aberta."""
        cinza = _para_cinza(frame)
        return {regiao: self.detectar_regiao(cinza, regiao) for regiao in REGIOES}

    def medir_regiao(self, frame: np.ndarray, regiao: str) -> dict:
        """Como detectar_regiao, mas devolve também as MEDIDAS internas (brilho
        e energia de alta frequência) usadas no cálculo da confiança. Serve para
        calibrar LIMIAR_PRETO / LIMIAR_ESTATICA olhando os números reais do jogo."""
        roi = recortar_roi(_para_cinza(frame), regiao)
        refs = self._refs.get(regiao)
        return {
            "perigo": self._perigo(roi, refs) if refs else 0.0,
            "conf":   self._confianca(roi),
            "brilho": float(roi.mean()),
            "std":    float(roi.std()),
            "hf":     float(cv2.Laplacian(roi, cv2.CV_64F).var()),
            "tem_ref": bool(refs),
        }

    def diff_regiao(self, frame: np.ndarray, regiao: str):
        """Diagnóstico: devolve (diff_map, stats) da região contra o vazio mais
        próximo, já alinhado. Para calibrar LIMIAR_DIFF olhando a diferença REAL —
        a sombra/rosto aparece no mapa? em que intensidade? que fração passa de
        cada limiar? `diff_map` é uint8 (tamanho do template)."""
        roi = recortar_roi(_para_cinza(frame), regiao)
        refs = self._refs.get(regiao)
        if not refs:
            return None, {}
        roi_b = _preparar(roi)
        m = int(TAM_CANONICO[0] * MARGEM_DESLIZE)
        melhor_diff = None
        melhor_frac = 2.0
        for ref in refs:
            ref_b = _preparar(ref)
            template = ref_b[m:-m, m:-m] if m > 0 else ref_b
            if template.size == 0 or template.shape[0] > roi_b.shape[0] or template.shape[1] > roi_b.shape[1]:
                template = ref_b
            resultado = cv2.matchTemplate(roi_b, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(resultado)
            x, y = max_loc
            recorte = roi_b[y:y + template.shape[0], x:x + template.shape[1]]
            diff = cv2.absdiff(recorte, template)
            frac = float((diff > LIMIAR_DIFF).mean())
            if frac < melhor_frac:
                melhor_frac, melhor_diff = frac, diff
        stats = {
            "brilho":   float(roi.mean()),
            "max_diff": float(melhor_diff.max()),
            "med_diff": float(melhor_diff.mean()),
            "f15":      float((melhor_diff > 15).mean()),
            "f25":      float((melhor_diff > 25).mean()),
            "f35":      float((melhor_diff > 35).mean()),
        }
        return melhor_diff, stats

    # ── núcleo ───────────────────────────────────────────────────────────────
    @staticmethod
    def _perigo(roi: np.ndarray, refs: list[np.ndarray]) -> float:
        """FRAÇÃO da região que mudou em relação ao vazio mais próximo.

        Para cada variante: alinha a referência (deslizando, p/ absorver o
        balanço/pan), tira a diferença pixel a pixel e mede que fração passou de
        LIMIAR_DIFF. Um intruso localizado (rosto na porta) vira uma área de
        mudança clara — sensível mesmo numa ROI com UI fixa, ao contrário da
        correlação global. Usa a variante de MENOR mudança (melhor casamento do
        vazio): se casa com qualquer vazio conhecido, perigo baixo."""
        roi_b = _preparar(roi)
        m = int(TAM_CANONICO[0] * MARGEM_DESLIZE)
        melhor = 1.0
        for ref in refs:
            ref_b = _preparar(ref)
            template = ref_b[m:-m, m:-m] if m > 0 else ref_b
            if template.size == 0 or template.shape[0] > roi_b.shape[0] or template.shape[1] > roi_b.shape[1]:
                template = ref_b
            resultado = cv2.matchTemplate(roi_b, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(resultado)
            x, y = max_loc
            recorte = roi_b[y:y + template.shape[0], x:x + template.shape[1]]
            diff = cv2.absdiff(recorte, template)
            fracao_mudou = float((diff > LIMIAR_DIFF).mean())
            melhor = min(melhor, fracao_mudou)
        return float(np.clip(melhor, 0.0, 1.0))

    @staticmethod
    def _confianca(roi: np.ndarray) -> float:
        """Baixa a confiança só quando a leitura não tem INFORMAÇÃO: quadro
        CHAPADO (preto/transição → desvio-padrão ~0) ou ESTÁTICA/glitch (alta
        energia de alta frequência). Cena escura mas com estrutura — típica do
        FNAF, brilho ~0 — é confiável: o CLAHE extrai o sinal mesmo assim. Por
        isso o critério é o desvio-padrão (há conteúdo?), não o brilho médio."""
        if float(roi.std()) < LIMIAR_FLAT:
            return CONF_PRETO

        hf = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        if hf > LIMIAR_ESTATICA:
            return CONF_ESTATICA

        return 1.0
