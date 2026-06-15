"""
Simulador de energia do FNAF1.

Dois modos:
  - realtempo  : cronômetro ao vivo — rode junto com o jogo e compare quando cada um zera
  - simulacao  : calcula o resultado instantaneamente, sem esperar

Uso:
  python -m src.utils.simular_energia              # modo real-tempo
  python -m src.utils.simular_energia simulacao    # modo instantâneo
"""

import sys
import time

PASSIVE  = 0.104   # %/s passivo base (Night 1, sem itens ativos)
PER_ITEM = 0.100   # %/s por item ativo adicional (porta, luz, câmera)

CHECKPOINTS = [
    ( 89,  85.0, "1 AM"),
    (178,  60.0, "2 AM"),
    (267,  40.0, "3 AM"),
    (356,  25.0, "4 AM"),
    (445,  15.0, "5 AM"),
    (535,   5.0, "6 AM"),
]


def _pedir_itens() -> int:
    print("=== Simulador de Energia FNAF1 ===")
    print(f"  Passivo base  : {PASSIVE:.3f} %/s")
    print(f"  Por item ativo: {PER_ITEM:.3f} %/s")
    print()
    print("Itens ativos que o jogo vai ter durante o teste (0-3):")
    print("  0 = só ventilador (nenhuma porta/luz/câmera)")
    print("  1 = + 1 item (ex: câmera aberta)")
    print("  2 = + 2 itens (ex: câmera + 1 porta)")
    print("  3 = + 3 itens (máximo)")
    while True:
        try:
            v = int(input("\nItens ativos: ").strip())
            if 0 <= v <= 3:
                return v
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Digite 0, 1, 2 ou 3.")


def _cabecalho_tabela():
    print(f"\n{'Tempo':>8} | {'Energia':>8} | {'Esperado':>9} | {'Diferença':>10}")
    print("-" * 46)


def _linha_checkpoint(elapsed: float, energia: float, esperado: float, nome: str):
    diff = energia - esperado
    sinal = "+" if diff >= 0 else ""
    print(f"{elapsed:7.1f}s | {energia:7.2f}% | {esperado:8.1f}% | {sinal}{diff:8.2f}%  ← {nome}")


def modo_simulacao(itens: int):
    consumo = PASSIVE + itens * PER_ITEM
    print(f"\nConsumo: {consumo:.3f} %/s")

    _cabecalho_tabela()

    energia = 100.0
    cp_idx = 0
    t = 0.0
    dt = 0.1  # resolução da simulação

    while energia > 0:
        t += dt
        energia = max(0.0, energia - consumo * dt)

        if cp_idx < len(CHECKPOINTS):
            t_cp, e_cp, nome_cp = CHECKPOINTS[cp_idx]
            if t >= t_cp:
                _linha_checkpoint(t, energia, e_cp, nome_cp)
                cp_idx += 1

    print(f"\n{'Energia zerou em:':>22} {t:.1f}s  ({t/60:.1f} min)")

    # Se acabou antes do fim da noite, diz quando
    if t < 535:
        print(f"{'Noite completa:':>22} 535.0s  ({535/60:.1f} min)")
        print(f"{'Diferença:':>22} -{535 - t:.1f}s antes do 6 AM")
    else:
        print(f"{'Chegou ao 6 AM com:':>22} {energia:.2f}% restantes")


def modo_realtempo(itens: int):
    consumo = PASSIVE + itens * PER_ITEM
    duracao_est = 100.0 / consumo

    print(f"\nConsumo: {consumo:.3f} %/s  |  Duração estimada: {duracao_est:.1f}s  ({duracao_est/60:.1f} min)")
    print("\nPressione Enter para iniciar (Ctrl+C para parar antecipadamente).")
    try:
        input()
    except KeyboardInterrupt:
        return

    energia = 100.0
    inicio = time.perf_counter()
    ultimo = inicio
    cp_idx = 0

    _cabecalho_tabela()

    try:
        while energia > 0:
            agora = time.perf_counter()
            dt = agora - ultimo
            ultimo = agora
            energia = max(0.0, energia - consumo * dt)
            elapsed = agora - inicio

            # Imprime checkpoint quando o tempo correspondente é atingido
            if cp_idx < len(CHECKPOINTS):
                t_cp, e_cp, nome_cp = CHECKPOINTS[cp_idx]
                if elapsed >= t_cp:
                    _linha_checkpoint(elapsed, energia, e_cp, nome_cp)
                    cp_idx += 1

            # Atualização ao vivo na mesma linha (a cada ~1s)
            print(f"\r  {elapsed:6.1f}s  |  {energia:5.2f}%  ", end="", flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - inicio
        print(f"\n\nInterrompido em {elapsed:.1f}s — energia restante: {energia:.2f}%")
        return

    elapsed = time.perf_counter() - inicio
    print(f"\n\nEnergia zerou em {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    if elapsed < 535:
        print(f"Antes do 6 AM: {535 - elapsed:.1f}s a menos que o necessário")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "realtempo"

    if modo not in ("realtempo", "simulacao"):
        print(f"Modo desconhecido: '{modo}'")
        print("Use: realtempo  ou  simulacao")
        sys.exit(1)

    itens = _pedir_itens()

    if modo == "simulacao":
        modo_simulacao(itens)
    else:
        modo_realtempo(itens)
