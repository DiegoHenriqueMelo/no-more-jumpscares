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
    import os, glob

    # Prioriza o modelo merged se existir
    if os.path.exists("pc1_fnaf_ppo_220359_steps.zip"):
        ultimo_modelo = "pc1_fnaf_ppo_220359_steps.zip"
        print("Usando modelo merged")
    else:
        modelos = glob.glob("modelos/*.zip")
        ultimo_modelo = max(modelos, key=os.path.getctime) if modelos else None

    if ultimo_modelo:
        print(f"Continuando treino: {ultimo_modelo}")
    else:
        print("Nenhum modelo encontrado — começando do zero")

    treinar(timesteps=500_000, carregar_modelo=ultimo_modelo)

if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "teste"

    if modo == "teste":
        modo_teste()
    elif modo == "treino":
        modo_treino()
    else:
        print(f"Modo desconhecido: {modo}")
        print("Use: python main.py teste | python main.py treino")