"""Mescla (média) os pesos de múltiplos modelos PPO treinados em PCs diferentes.

Equivale ao Federated Learning simplificado: combina o aprendizado de vários
agentes treinados independentemente em suas próprias máquinas.

Uso
----
    python merge_modelos.py modelo1.zip modelo2.zip [modelo3.zip ...]

Exemplo com 3 PCs:
    python merge_modelos.py modelos/pc1_20k.zip modelos/pc2_20k.zip modelos/pc3_20k.zip

O modelo resultante é salvo em modelos/fnaf_merged.zip e pode ser usado como
ponto de partida para continuar o treino ou para transferência de pesos para
RecurrentPPO:
    python -m src.agent.train_recurrent --ppo-antigo modelos/fnaf_merged.zip
"""
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.utils.dummy_env import DummyFNAFEnv


def merge_modelos(caminhos: list[str], saida: str = "modelos/fnaf_merged.zip"):
    """Faz a média dos pesos de múltiplos modelos PPO.

    Args:
        caminhos: lista de caminhos para modelos PPO (.zip).
        saida: caminho onde salvar o modelo mesclado.
    """
    print(f"Carregando {len(caminhos)} modelos...")

    # DummyFNAFEnv: carrega os modelos sem precisar abrir o jogo
    env = DummyVecEnv([DummyFNAFEnv])

    modelo_base = PPO.load(caminhos[0], env=env)
    params_base = modelo_base.policy.state_dict()
    print(f"  [1/{len(caminhos)}] {caminhos[0]} carregado")

    for i, caminho in enumerate(caminhos[1:], start=2):
        modelo = PPO.load(caminho, env=env)
        params = modelo.policy.state_dict()
        for chave in params_base:
            params_base[chave] = params_base[chave] + params[chave]
        print(f"  [{i}/{len(caminhos)}] {caminho} carregado")

    for chave in params_base:
        params_base[chave] = params_base[chave] / len(caminhos)

    modelo_base.policy.load_state_dict(params_base)
    modelo_base.save(saida)

    print(f"\nModelo merged salvo em: {saida}")
    print(f"Equivalente a {len(caminhos)}x mais experiência combinada!")

    env.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python merge_modelos.py modelo1.zip modelo2.zip [modelo3.zip ...]")
        print("Exemplo: python merge_modelos.py modelos/pc1_20k.zip modelos/pc2_20k.zip modelos/pc3_20k.zip")
        sys.exit(1)

    merge_modelos(sys.argv[1:])
