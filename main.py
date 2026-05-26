"""Ponto de entrada principal do projeto No More Jumpscares.

Modos disponíveis
-----------------
    python main.py teste          # testa reset do ambiente
    python main.py treino         # treino RecurrentPPO com LSTM (recomendado)
    python main.py treino-legacy  # treino PPO padrão (sem LSTM, legado)

Para opções avançadas (timesteps, modelo, ppo-antigo, etc.) use os scripts
diretamente:
    python -m src.agent.train_recurrent --help
    python -m src.agent.train --help
"""
import sys
from src.environment.fnaf_env import FNAFEnv


def modo_teste():
    print("Testando reset...")
    env = FNAFEnv()
    obs, _ = env.reset()
    print(f"Reset OK! Imagem: {obs['imagem'].shape}, Estados: {obs['estados'].shape}")
    print(f"Estados iniciais:")
    print(f"  - Porta esquerda: {obs['estados'][0]}")
    print(f"  - Porta direita: {obs['estados'][1]}")
    print(f"  - Luz esquerda: {obs['estados'][2]}")
    print(f"  - Luz direita: {obs['estados'][3]}")
    print(f"  - Câmera aberta: {obs['estados'][4]}")
    print(f"  - Câmera ativa: {obs['estados'][5]:.2f}")
    print(f"  - Energia: {obs['estados'][6]*100:.1f}%")
    input("O jogo iniciou a noite 1? (aperta Enter para confirmar)")
    env.close()


def modo_treino():
    """Treino PPO padrão (legado). Prefira modo_treino_recurrent."""
    from src.agent.train import treinar
    import os, glob, re

    merged = glob.glob("modelos/*merged*.zip")
    if merged:
        ultimo_modelo = max(merged, key=os.path.getctime)
        print(f"Usando modelo merged: {ultimo_modelo}")
    else:
        def extrair_steps(path):
            m = re.search(r"_(\d+)_steps\.zip$", path)
            return int(m.group(1)) if m else -1

        numerados = [p for p in glob.glob("modelos/*.zip") if extrair_steps(p) >= 0]
        if numerados:
            ultimo_modelo = max(numerados, key=extrair_steps)
            print(f"Continuando treino: {ultimo_modelo} ({extrair_steps(ultimo_modelo):,} steps)")
        else:
            outros = glob.glob("modelos/*.zip")
            ultimo_modelo = max(outros, key=os.path.getctime) if outros else None
            if ultimo_modelo:
                print(f"Continuando treino (sem steps no nome): {ultimo_modelo}")
            else:
                print("Nenhum modelo encontrado — começando do zero")

    treinar(timesteps=500_000, carregar_modelo=ultimo_modelo)


def modo_treino_recurrent():
    """Treino RecurrentPPO com LSTM — algoritmo principal do projeto."""
    from src.agent.train_recurrent import treinar
    import os, glob, re

    # Procura modelos RecurrentPPO salvos anteriormente
    recurrents = glob.glob("modelos/*recurrent*.zip")
    ultimo_modelo = None

    if recurrents:
        def extrair_steps(path):
            m = re.search(r"_(\d+)_steps\.zip$", path)
            return int(m.group(1)) if m else -1

        numerados = [p for p in recurrents if extrair_steps(p) >= 0]
        if numerados:
            ultimo_modelo = max(numerados, key=extrair_steps)
            print(f"Continuando treino RecurrentPPO: {ultimo_modelo} ({extrair_steps(ultimo_modelo):,} steps)")
        else:
            ultimo_modelo = max(recurrents, key=os.path.getctime)
            print(f"Continuando treino RecurrentPPO: {ultimo_modelo}")

    if not ultimo_modelo:
        # Procura modelos BC para warmup
        bc = "modelos/fnaf_bc.zip"
        ppo = "modelos/fnaf_ppo_final.zip"
        if os.path.exists(bc):
            print(f"Nenhum RecurrentPPO encontrado — iniciando com warmup BC: {bc}")
            treinar(timesteps=500_000, carregar_ppo_antigo=bc)
        elif os.path.exists(ppo):
            print(f"Nenhum RecurrentPPO encontrado — transferindo pesos do PPO: {ppo}")
            treinar(timesteps=500_000, carregar_ppo_antigo=ppo)
        else:
            print("Nenhum modelo encontrado — começando do zero")
            treinar(timesteps=500_000)
    else:
        treinar(timesteps=500_000, carregar_modelo=ultimo_modelo)


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "teste"

    if modo == "teste":
        modo_teste()
    elif modo == "treino":
        modo_treino_recurrent()
    elif modo in ("treino-legacy", "treino_legacy", "legacy"):
        modo_treino()
    elif modo in ("treino-recurrent", "treino_recurrent", "recurrent"):
        # Alias mantido para compatibilidade
        modo_treino_recurrent()
    else:
        print(f"Modo desconhecido: {modo}")
        print("Use:")
        print("  python main.py teste          — testa o reset do ambiente")
        print("  python main.py treino         — treino RecurrentPPO (recomendado)")
        print("  python main.py treino-legacy  — treino PPO padrão (sem LSTM, legado)")
