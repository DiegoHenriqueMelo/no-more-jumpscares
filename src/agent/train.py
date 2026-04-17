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

def _carregar_env(caminho: str = ".env") -> None:
    if not os.path.exists(caminho):
        return

    with open(caminho, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            conteudo = linha.strip()
            if not conteudo or conteudo.startswith("#") or "=" not in conteudo:
                continue

            chave, valor = conteudo.split("=", 1)
            chave = chave.strip()
            valor = valor.strip().strip('"').strip("'")
            os.environ.setdefault(chave, valor)

def _env_str_obrigatorio(nome: str) -> str:
    valor = os.getenv(nome)
    if valor is None or valor.strip() == "":
        raise ValueError(f"Variavel obrigatoria ausente no .env: {nome}")
    return valor.strip()


class LogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodio         = 0
        self.mortes           = 0
        self.vitorias         = 0
        self.recompensa_total = 0.0
        self.inicio_ep        = None

        os.makedirs("logs", exist_ok=True)
        self.arquivo_log = open("logs/treino.log", "a", encoding="utf-8")
        self.arquivo_log.write(f"\n{'='*60}\nTreino iniciado\n{'='*60}\n")

    def _on_step(self) -> bool:
        # F12 pausa a IA — segura para pausar, larga para continuar
        while keyboard.is_pressed("F12"):
            print("PAUSADO — solte F12 para continuar...", end="\r")
            time.sleep(0.5)

        if self.inicio_ep is None:
            # Marca o inicio real do episodio para calcular duracao em minutos.
            self.inicio_ep = time.perf_counter()

        info = self.locals.get("infos", [{}])[0]
        self.recompensa_total += self.locals.get("rewards", [0])[0]

        done = self.locals.get("dones", [False])[0]
        if done:
            agora = time.perf_counter()
            tempo_ep_minutos = (
                (agora - self.inicio_ep) / 60.0
                if self.inicio_ep is not None
                else 0.0
            )

            self.episodio += 1

            if info.get("morreu", False):
                self.mortes += 1
                resultado = "MORTE"
            else:
                self.vitorias += 1
                resultado = "VITORIA"

            taxa_vitoria = (self.vitorias / self.episodio) * 100

            linha = (
                f"{_env_str_obrigatorio('PC')} | "
                f"Ep {self.episodio:4d} | "
                f"{resultado:8s} | "
                f"Passos: {info.get('passos', 0):6d} | "
                f"Tempo: {tempo_ep_minutos:7.2f} min | "
                f"Recompensa: {self.recompensa_total:8.1f} | "
                f"Taxa vitória: {taxa_vitoria:.1f}%"
            )

            print(linha)
            self.arquivo_log.write(linha + "\n")
            self.arquivo_log.flush()
            self.recompensa_total = 0.0
            self.inicio_ep = None

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
        name_prefix=f"{_env_str_obrigatorio('PC')}_fnaf_ppo",
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