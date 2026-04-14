import os
import time
import keyboard
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from src.environment.fnaf_env import FNAFEnv

PASTA_MODELOS = "modelos"
PASTA_LOGS    = "logs"
os.makedirs(PASTA_MODELOS, exist_ok=True)
os.makedirs(PASTA_LOGS,    exist_ok=True)


class LogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodio         = 0
        self.mortes           = 0
        self.vitorias         = 0
        self.recompensa_total = 0.0

        os.makedirs("logs", exist_ok=True)
        self.arquivo_log = open("logs/treino.log", "a", encoding="utf-8")
        self.arquivo_log.write(f"\n{'='*60}\nTreino iniciado\n{'='*60}\n")

    def _on_step(self) -> bool:
        # F12 pausa a IA — segura para pausar, larga para continuar
        while keyboard.is_pressed("F12"):
            print("PAUSADO — solte F12 para continuar...", end="\r")
            time.sleep(0.5)

        info = self.locals.get("infos", [{}])[0]
        self.recompensa_total += self.locals.get("rewards", [0])[0]

        done = self.locals.get("dones", [False])[0]
        if done:
            self.episodio += 1

            if info.get("morreu", False):
                self.mortes += 1
                resultado = "MORTE"
            else:
                self.vitorias += 1
                resultado = "VITORIA"

            taxa_vitoria = (self.vitorias / self.episodio) * 100

            linha = (
                f"Ep {self.episodio:4d} | "
                f"{resultado:8s} | "
                f"Passos: {info.get('passos', 0):6d} | "
                f"Recompensa: {self.recompensa_total:8.1f} | "
                f"Taxa vitória: {taxa_vitoria:.1f}%"
            )

            print(linha)
            self.arquivo_log.write(linha + "\n")
            self.arquivo_log.flush()
            self.recompensa_total = 0.0

        return True

    def _on_training_end(self):
        self.arquivo_log.write("Treino finalizado\n")
        self.arquivo_log.close()


def treinar(timesteps: int = 500_000, carregar_modelo: str = None):
    print("Iniciando ambiente FNAF1...")
    print("ATENÇÃO: Deixe o jogo aberto e na tela inicial!")
    print("Dica: segure F12 a qualquer momento para pausar.\n")
    time.sleep(3)

    env = FNAFEnv()

    if carregar_modelo and os.path.exists(carregar_modelo):
        print(f"Carregando modelo: {carregar_modelo}")
        modelo = PPO.load(carregar_modelo, env=env)
    else:
        print("Criando novo modelo PPO...")
        modelo = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=0,
            tensorboard_log=PASTA_LOGS,
            device="auto",
        )

    checkpoint = CheckpointCallback(
        save_freq=10_000,
        save_path=PASTA_MODELOS,
        name_prefix="fnaf_ppo",
    )

    print(f"Treinando por {timesteps:,} timesteps...\n")
    modelo.learn(
        total_timesteps=timesteps,
        callback=[checkpoint, LogCallback()],
        reset_num_timesteps=carregar_modelo is None,
    )

    caminho_final = f"{PASTA_MODELOS}/fnaf_ppo_final"
    modelo.save(caminho_final)
    print(f"\nModelo final salvo em: {caminho_final}")

    env.close()


if __name__ == "__main__":
    treinar(
        timesteps=500_000,
        carregar_modelo=None,
    )