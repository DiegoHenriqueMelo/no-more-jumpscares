"""Ambiente FNAF mínimo para carregar modelos sem abrir o jogo.

Usado por behavioral_cloning.py, train_recurrent.py e merge_modelos.py quando
é necessário instanciar PPO/RecurrentPPO sem conectar ao jogo real (ex: carregar
pesos, fazer transferência de features, mesclar modelos).

    from src.utils.dummy_env import DummyFNAFEnv
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([DummyFNAFEnv])
    modelo = PPO.load("modelos/fnaf_ppo_final.zip", env=env)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ALTURA    = 84
LARGURA   = 84
NUM_ACOES = 17
# Dimensão do vetor de estados (deve ser igual ao FNAFEnv.observation_space["estados"])
NUM_ESTADOS = 9


class DummyFNAFEnv(gym.Env):
    """Ambiente mínimo compatível com gym para criar PPO/RecurrentPPO sem abrir o jogo.

    Tem o mesmo observation_space e action_space que FNAFEnv, mas todos os métodos
    retornam observações zeradas. Não carrega templates, não lê janelas, não chama
    pyautogui — é seguro usar em contextos sem jogo rodando.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({
            "imagem":  spaces.Box(low=0, high=255, shape=(ALTURA, LARGURA, 1), dtype=np.uint8),
            "estados": spaces.Box(low=0, high=1,   shape=(NUM_ESTADOS,),       dtype=np.float32),
        })
        self.action_space = spaces.Discrete(NUM_ACOES)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs_zero(), {}

    def step(self, action):
        return self._obs_zero(), 0.0, True, False, {}

    def _obs_zero(self):
        return {
            "imagem":  np.zeros((ALTURA, LARGURA, 1), dtype=np.uint8),
            "estados": np.zeros(NUM_ESTADOS, dtype=np.float32),
        }

    def close(self):
        pass
