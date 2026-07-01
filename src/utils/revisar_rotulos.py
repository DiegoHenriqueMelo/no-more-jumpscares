"""
Revisa e (re)rotula as amostras de detecção OLHANDO cada frame — o jeito
confiável de rotular. A captura em lote (rotular_deteccao) erra quando o
animatrônico aparece no meio da sequência; aqui você corrige frame a frame.

    python -m src.utils.revisar_rotulos [regiao]

Sem <regiao>, revisa todas as que têm amostras. Mostra cada frame com a ROI
desenhada. Teclas:
    v = marcar VAZIO     c = marcar CHEIO
    d = apagar           s = pular (mantém)      q = sair

Método: o perigo do detector NÃO é mostrado de propósito — sua label deve ser
ground-truth independente (senão a avaliação vira circular).
"""
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.environment.deteccao_visual import REGIOES, roi_da_regiao

DEST = Path("dados/rotulos_deteccao")


def _renomear(f: Path, novo_rotulo: str) -> Path:
    """Renomeia <rotuloantigo>_<ts>.png para <novo_rotulo>_<ts>.png."""
    ts = f.name.split("_", 1)[1] if "_" in f.name else f.name
    destino = f.with_name(f"{novo_rotulo}_{ts}")
    if destino != f:
        if destino.exists():
            destino.unlink()
        f.rename(destino)
    return destino


def _revisar(regiao: str, arquivos: list) -> None:
    i = 0
    while i < len(arquivos):
        f = arquivos[i]
        if not f.exists():
            i += 1
            continue
        frame = cv2.imread(str(f))
        if frame is None:
            i += 1
            continue

        rotulo_atual = "cheio" if f.name.startswith("cheio") else "vazio"
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = roi_da_regiao(regiao)
        vis = frame.copy()
        cv2.rectangle(vis, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (0, 255, 0), 2)
        cv2.putText(vis, f"{regiao} [{i + 1}/{len(arquivos)}]  rotulo atual: {rotulo_atual}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, "v=vazio  c=cheio  d=apagar  s=pular  q=sair",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        escala = min(1.0, 1100.0 / w)
        if escala < 1.0:
            vis = cv2.resize(vis, (int(w * escala), int(h * escala)))

        cv2.imshow("revisar rotulos", vis)
        k = cv2.waitKey(0) & 0xFF

        if k == ord("q"):
            break
        elif k == ord("v"):
            _renomear(f, "vazio"); i += 1
        elif k == ord("c"):
            _renomear(f, "cheio"); i += 1
        elif k == ord("d"):
            f.unlink(); i += 1
        elif k == ord("s"):
            i += 1
        # qualquer outra tecla: re-exibe o mesmo frame

    cv2.destroyAllWindows()


def main():
    if not DEST.exists():
        print(f"Sem amostras em {DEST}. Rode rotular_deteccao primeiro.")
        return

    if len(sys.argv) > 1 and sys.argv[1] in REGIOES:
        regioes = [sys.argv[1]]
    else:
        regioes = [r for r in REGIOES if (DEST / r).exists()]

    for regiao in regioes:
        pasta = DEST / regiao
        arquivos = sorted(pasta.glob("*.png")) if pasta.exists() else []
        if not arquivos:
            continue
        print(f"Revisando {regiao}: {len(arquivos)} frames "
              f"(v=vazio c=cheio d=apagar s=pular q=sair)")
        _revisar(regiao, arquivos)

    print("\nResumo:")
    for regiao in regioes:
        pasta = DEST / regiao
        if pasta.exists():
            nv = len(list(pasta.glob("vazio_*.png")))
            nc = len(list(pasta.glob("cheio_*.png")))
            print(f"  {regiao}: vazio={nv}  cheio={nc}")
    print("Agora avalie: python -m src.utils.avaliar_deteccao")


if __name__ == "__main__":
    main()
