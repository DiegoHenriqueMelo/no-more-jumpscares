"""
Avalia a PRECISÃO da detecção sobre as amostras rotuladas (rotular_deteccao).

Por região, trata o perigo como um classificador binário (tem animatrônico?) e
imprime: distribuição de perigo (vazio vs cheio), o melhor limiar de separação,
acurácia/precisão/recall e — o mais importante — se as classes se SEPARAM com
folga. É o número que diz se dá pra confiar na detecção, em vez do olho.

    python -m src.utils.avaliar_deteccao

OBS: o agente recebe o perigo CONTÍNUO (não um limiar). O limiar aqui é só para
medir a separabilidade. Se 'vazio' e 'cheio' se sobrepõem, a detecção daquela
região está imprecisa — aí ajustamos ROI / LIMIAR_DIFF / referências.
"""
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.environment.deteccao_visual import DetectorAnimatronicos, REGIOES, CONF_GATE

DEST = Path("dados/rotulos_deteccao")


def _avaliar_regiao(det: DetectorAnimatronicos, regiao: str, arquivos: list[Path]) -> None:
    amostras = []  # (label, perigo, conf)
    for f in arquivos:
        rotulo = 1 if f.name.startswith("cheio") else 0
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        perigo, conf = det.detectar_regiao(frame, regiao)
        amostras.append((rotulo, perigo, conf))

    vazios = [p for r, p, c in amostras if r == 0]
    cheios = [p for r, p, c in amostras if r == 1]
    if not vazios or not cheios:
        print(f"\n=== {regiao} ===  (vazio={len(vazios)}, cheio={len(cheios)}) — "
              f"precisa de amostras dos DOIS rótulos; pulei.")
        return

    # Melhor limiar por acurácia (varre 0..1)
    melhor_thr, melhor_acc = 0.0, 0.0
    for i in range(0, 101, 2):
        thr = i / 100.0
        tp = sum(p >= thr for p in cheios)
        tn = sum(p < thr for p in vazios)
        acc = (tp + tn) / (len(cheios) + len(vazios))
        if acc > melhor_acc:
            melhor_acc, melhor_thr = acc, thr

    thr = melhor_thr
    tp = sum(p >= thr for p in cheios)
    fn = len(cheios) - tp
    fp = sum(p >= thr for p in vazios)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    sep = min(cheios) - max(vazios)  # >0 = gap limpo entre as classes
    baixa_conf = sum(c < CONF_GATE for r, p, c in amostras)

    print(f"\n=== {regiao} ===  (vazio={len(vazios)}, cheio={len(cheios)})")
    print(f"  perigo vazio: media {sum(vazios)/len(vazios):.2f}  max {max(vazios):.2f}")
    print(f"  perigo cheio: media {sum(cheios)/len(cheios):.2f}  min {min(cheios):.2f}")
    print(f"  limiar otimo {thr:.2f} -> acuracia {melhor_acc*100:3.0f}%  "
          f"precisao {prec*100:3.0f}%  recall {rec*100:3.0f}%")
    if sep > 0:
        print(f"  SEPARAVEL (folga {sep:.2f} entre max-vazio e min-cheio) -> confiavel")
    else:
        print(f"  SOBREPOSTO (vazio chega a {max(vazios):.2f}, cheio cai a {min(cheios):.2f}) "
              f"-> impreciso, ajustar")
    if baixa_conf:
        print(f"  [aviso] {baixa_conf}/{len(amostras)} amostras com conf < {CONF_GATE} "
              f"(o env descartaria essas leituras)")


def main():
    if not DEST.exists():
        print(f"Sem amostras em {DEST}. Rode primeiro: python -m src.utils.rotular_deteccao <regiao> <vazio|cheio>")
        return

    det = DetectorAnimatronicos()
    achou = False
    for regiao in REGIOES:
        pasta = DEST / regiao
        if not pasta.exists():
            continue
        arquivos = sorted(pasta.glob("*.png"))
        if arquivos:
            achou = True
            _avaliar_regiao(det, regiao, arquivos)

    if not achou:
        print(f"Nenhuma amostra encontrada em {DEST}/<regiao>/.")
    else:
        print("\nLeitura: SEPARAVEL = perigo do vazio e do cheio não se cruzam (bom).")
        print("         SOBREPOSTO = há frames vazios com perigo tão alto quanto cheios (ruim).")


if __name__ == "__main__":
    main()
