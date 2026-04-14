import sys
from src.environment.fnaf_env import FNAFEnv

def modo_teste():
    print("Testando reset...")
    env = FNAFEnv()
    obs, info = env.reset()
    print(f"Reset OK! Shape: {obs.shape}")
    input("O jogo iniciou a noite 1? (aperta Enter para confirmar)")
    env.close()

def modo_treino():
    from src.agent.train import treinar
    treinar(timesteps=500_000)

if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "teste"

    if modo == "teste":
        modo_teste()
    elif modo == "treino":
        modo_treino()
    else:
        print(f"Modo desconhecido: {modo}")
        print("Use: python main.py teste | python main.py treino")