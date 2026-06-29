"""
Diagnóstico da detecção de ocupação (animatrônicos).

IMPORTANTE: este script é "cego" ao estado do jogo — ele lê TODAS as regiões a
cada frame. Mas uma região só faz sentido quando você está observando ela:
  - câmera  → só vale com a TAB daquela câmera aberta;
  - porta/janela → só vale no escritório, com a LUZ daquele lado acesa.
Na integração com o RL isso é resolvido pelo env, que só lê a região observável.

Modos:

  # ao vivo, todas as regiões (mostra perigo/conf de cada uma):
  python -m src.utils.testar_deteccao_animatronicos

  # FOCO numa região — mostra perigo, conf, brilho e ruído (hf) ao vivo.
  # Use para validar uma de cada vez E para calibrar os limiares de confiança:
  python -m src.utils.testar_deteccao_animatronicos porta_esq

  # snapshot: congela 1 frame e salva o que cada ROI vê vs a referência do vazio:
  python -m src.utils.testar_deteccao_animatronicos snapshot

Ctrl+C para sair.
"""
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.environment.deteccao_visual import (
    DetectorAnimatronicos,
    REGIOES,
    roi_da_regiao,
    recortar_roi,
    TAM_CANONICO,
    LIMIAR_FLAT,
    LIMIAR_ESTATICA,
    CONF_GATE,
)
from src.utils.capture import GameCapture

DEBUG_DIR = Path("dados/debug_deteccao")


def _capturar_janela(cap: GameCapture):
    import pygetwindow as gw
    from src.environment.fnaf_env import WINDOW_TITLE

    if not WINDOW_TITLE:
        return None
    janelas = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not janelas:
        return None
    win = janelas[0]
    regiao = {"left": win.left, "top": win.top, "width": win.width, "height": win.height}
    return cap.capturar_tela(regiao)


def modo_foco(cap, detector, regiao: str):
    """Leitura detalhada de UMA região: perigo, conf e as medidas que decidem a
    confiança (brilho e ruído). É aqui que se calibra LIMIAR_PRETO/LIMIAR_ESTATICA."""
    if regiao not in REGIOES:
        print(f"Região inválida: {regiao}. Use: {', '.join(REGIOES)}")
        return

    print(f"FOCO: {regiao}")
    print(f"Limiares atuais: LIMIAR_FLAT={LIMIAR_FLAT} (std menor = 'chapado/sem info'), "
          f"LIMIAR_ESTATICA={LIMIAR_ESTATICA} (hf maior = 'estática')")
    print("Compare VAZIO vs ANIMATRÔNICO: o 'perigo' tem que subir. "
          "Cena escura com std > LIMIAR_FLAT é confiável.\n")

    try:
        while True:
            frame = _capturar_janela(cap)
            if frame is None:
                print("Jogo não encontrado — aguardando...", end="\r")
                time.sleep(0.5)
                continue

            m = detector.medir_regiao(frame, regiao)
            if not m["tem_ref"]:
                motivo = "(sem ref — capture com 'calibrar vazio')"
            elif m["std"] < LIMIAR_FLAT:
                motivo = "<- conf baixa: CHAPADO (std < LIMIAR_FLAT, sem info)"
            elif m["hf"] > LIMIAR_ESTATICA:
                motivo = "<- conf baixa: ESTÁTICA (hf > LIMIAR_ESTATICA)"
            else:
                motivo = "ok"
            print(f"  perigo={m['perigo']:.2f}  conf={m['conf']:.2f}  "
                  f"brilho={m['brilho']:5.1f}  std={m['std']:5.1f}  hf={m['hf']:7.1f}   {motivo}      ", end="\r")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nEncerrado.")


def modo_todas(cap, detector):
    faltando = detector.regioes_faltando()
    if faltando:
        print(f"[AVISO] Sem referência: {', '.join(faltando)} (capture com 'calibrar vazio')")
    print("Formato: perigo/conf  | só confie na região que você observa agora.\n")

    cabecalho = " | ".join(f"{r:>11}" for r in REGIOES)
    print(f"  {cabecalho}")
    print("-" * (len(cabecalho) + 4))

    try:
        while True:
            frame = _capturar_janela(cap)
            if frame is None:
                print("Jogo não encontrado — aguardando...", end="\r")
                time.sleep(0.5)
                continue

            celulas = []
            for r in REGIOES:
                if r in faltando:
                    celulas.append("(sem ref)")
                else:
                    perigo, conf = detector.detectar_regiao(frame, r)
                    celulas.append(f"{perigo:.2f}/{conf:.2f}")
            print("  " + " | ".join(f"{c:>11}" for c in celulas), end="\r")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nEncerrado.")


def modo_snapshot(cap, detector):
    """Congela 1 frame e salva, por região: [o que a ROI vê | referência do vazio]
    + o frame inteiro com as ROIs desenhadas + as medidas (perigo/conf/brilho/hf)."""
    print("SNAPSHOT — deixe a cena alvo na tela (ex.: animatrônico na porta).")
    print("Capturando em 5 segundos...")
    time.sleep(5)

    frame = _capturar_janela(cap)
    if frame is None:
        print("Jogo não encontrado.")
        return

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    anotado = frame.copy()
    h, w = frame.shape[:2]

    print(f"\n{'região':>10} | perigo | brilho | maxdiff | f>15  f>25  f>35")
    print("-" * 66)
    for regiao in REGIOES:
        m = detector.medir_regiao(frame, regiao)
        roi_atual = recortar_roi(cinza, regiao)

        refs = detector._refs.get(regiao)
        ref_img = refs[0] if refs else (roi_atual * 0)

        # mapa de diferença (alinhado) amplificado p/ enxergar a forma do intruso
        diff_map, dstats = detector.diff_regiao(frame, regiao)
        if diff_map is not None:
            diff_vis = cv2.resize(diff_map, TAM_CANONICO)
            diff_vis = cv2.convertScaleAbs(diff_vis, alpha=3.0)  # ×3 p/ visibilidade
        else:
            diff_vis = roi_atual * 0
        montagem = cv2.hconcat([roi_atual, ref_img, diff_vis])  # [atual | vazio | diff×3]
        cv2.imwrite(str(DEBUG_DIR / f"{regiao}.png"), montagem)

        x0f, y0f, x1f, y1f = roi_da_regiao(regiao)
        p1 = (int(x0f * w), int(y0f * h))
        p2 = (int(x1f * w), int(y1f * h))
        cor = (0, 0, 255) if (m["conf"] >= CONF_GATE and m["perigo"] > 0.3) else (0, 255, 0)
        cv2.rectangle(anotado, p1, p2, cor, 2)
        cv2.putText(anotado, regiao, (p1[0] + 3, p1[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)

        if not refs:
            print(f"{regiao:>10} |  ---   |  ---   |  (sem referência)")
        else:
            chapado = " CHAPADO" if m["std"] < LIMIAR_FLAT else ""
            print(f"{regiao:>10} | {m['perigo']:5.2f}  | {m['brilho']:6.1f} | "
                  f"{dstats['max_diff']:6.0f}  | {dstats['f15']:4.2f}  "
                  f"{dstats['f25']:4.2f}  {dstats['f35']:4.2f}{chapado}")

    cv2.imwrite(str(DEBUG_DIR / "_frame.png"), anotado)
    print(f"\nSalvo em {DEBUG_DIR}/ :")
    print("  _frame.png    → frame inteiro com as ROIs desenhadas")
    print("  <regiao>.png  → [atual | vazio | mapa de diferença ×3]")
    print("No 'diff' (3º painel): se a forma do animatrônico APARECE mas f>35 é baixo")
    print("→ baixamos LIMIAR_DIFF. Se o diff é todo preto → ROI/alinhamento errado.")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    cap = GameCapture()
    detector = DetectorAnimatronicos()

    if arg == "snapshot":
        modo_snapshot(cap, detector)
    elif arg:
        modo_foco(cap, detector, arg)
    else:
        modo_todas(cap, detector)


if __name__ == "__main__":
    main()
