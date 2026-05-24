"""Treinamento com RecurrentPPO (LSTM) para o ambiente FNAF.

Por que RecurrentPPO em vez de PPO padrão
------------------------------------------
O PPO padrão toma decisões com base em um único frame (84×84px) + 8 estados
instantâneos. Ele não tem memória: cada step é tratado como se fosse o início
da partida. No FNAF, isso é um problema estrutural — os animatrônicos se movem
quando não estão sendo observados. Sem memória, o agente não consegue raciocinar
sobre "eu vi Bonnie na 2A há 5 steps, preciso fechar a porta esquerda".

O RecurrentPPO insere uma camada LSTM entre o extrator de features e as cabeças
de ator/crítico. O LSTM mantém um estado oculto (h, c) que persiste entre os
steps do episódio e é zerado no início de cada episódio via episode_start masks.

Arquitetura:
    Obs (frame + estados) → MultimodalExtractor (CNN + MLP → 256)
        → LSTM (256 → 256, 1 camada, compartilhada ator/crítico)
            → Ator: distribuição sobre 17 ações
            → Crítico: estimativa de V(estado)

Paralelismo entre máquinas
---------------------------
O FNAF usa pyautogui para clicar — compartilha o cursor do OS. Isso impede
múltiplas instâncias do jogo no mesmo PC (dois processos brigariam pelo cursor).
O paralelismo real é rodar um agente por PC, todos gravando no mesmo MongoDB,
e combinar os modelos periodicamente com merge_modelos.py.

Transferência de pesos do PPO antigo
--------------------------------------
Se houver um modelo PPO treinado anteriormente, os pesos do extrator de features
(CNN + MLP) podem ser transferidos para o RecurrentPPO. Isso preserva o que o
agente já aprendeu sobre a aparência do jogo. Use o parâmetro:
    --ppo-antigo modelos/fnaf_ppo_final.zip
"""
import os
import sys
import time
import keyboard
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.fnaf_env import FNAFEnv
from src.agent.multimodal_policy import MultimodalExtractor
from src.utils.dummy_env import DummyFNAFEnv

PASTA_MODELOS = "modelos"
PASTA_LOGS    = "logs"


def make_env_fn():
    """Retorna uma função (thunk) que cria um FNAFEnv.

    Usado com DummyVecEnv, que exige uma lista de callables:
        env = DummyVecEnv([make_env_fn()])
    """
    def _init():
        return FNAFEnv()
    return _init


class LogCallback(BaseCallback):
    """Loga progresso do treinamento step a step e por episódio."""

    def __init__(self, log_steps: bool = False):
        super().__init__()
        self.episodio          = 0
        self.episodios_validos = 0
        self.mortes            = 0
        self.vitorias          = 0
        self.interrompidos     = 0
        self._recompensa_ep    = 0.0
        self._pausa_disponivel = True
        self._log_steps        = log_steps
        self._pc               = os.getenv("PC", "")

        os.makedirs("logs", exist_ok=True)
        cabecalho = f"\n{'='*60}\nTreino RecurrentPPO (LSTM) iniciado\n{'='*60}\n"

        self.arquivo_log = open("logs/treino_recurrent.log", "a", encoding="utf-8")
        self.arquivo_log.write(cabecalho)

        self.arquivo_log_steps = None
        if log_steps:
            self.arquivo_log_steps = open("logs/treino_recurrent_steps.log", "a", encoding="utf-8")
            self.arquivo_log_steps.write(cabecalho)

    def _on_step(self) -> bool:
        # F12 pausa a IA — segura para pausar, larga para continuar
        if self._pausa_disponivel:
            try:
                while keyboard.is_pressed("F12"):
                    print("PAUSADO — solte F12 para continuar...", end="\r")
                    time.sleep(0.5)
            except Exception as erro:
                print(f"Aviso: pausa por F12 desativada. Motivo: {erro}")
                self._pausa_disponivel = False

        infos   = self.locals.get("infos",   [{}])
        rewards = self.locals.get("rewards", [0.0])
        dones   = self.locals.get("dones",   [False])
        info, reward, done = infos[0], rewards[0], dones[0]

        self._recompensa_ep += reward

        energia = info.get("energia")
        if energia is not None and self._log_steps:
            pe    = int(info.get("porta_esq",     False))
            pd    = int(info.get("porta_dir",     False))
            le    = int(info.get("luz_esq",       False))
            ld    = int(info.get("luz_dir",       False))
            ca    = int(info.get("camera_aberta", False))
            cv    = int(info.get("camera_ativa",  0))
            acao  = info.get("acao_nome", "?")
            valida = "OK" if info.get("acao_valida", True) else "X "
            linha_step = (
                f"{self._pc} | "
                f"Ep {self.episodio:4d} | "
                f"E:{energia:5.1f}% | "
                f"PE:{pe} PD:{pd} LE:{le} LD:{ld} | "
                f"CAM:{ca}/{cv:2d} | "
                f"#{info.get('passos', 0):5d} | "
                f"{acao:<20} [{valida}]"
            )
            print(linha_step)
            if self.arquivo_log_steps:
                self.arquivo_log_steps.write(linha_step + "\n")
                self.arquivo_log_steps.flush()

        if done:
            tempo_ep_minutos = info.get("tempo_real", 0.0) / 60.0
            recompensa_ep    = self._recompensa_ep
            self._recompensa_ep = 0.0

            self.episodio += 1
            interrompido = info.get("interrompido", False)

            if interrompido:
                self.interrompidos += 1
                resultado = "INTERROMPIDO"
            elif info.get("morreu", False):
                self.episodios_validos += 1
                self.mortes += 1
                resultado = "MORTE"
            else:
                self.episodios_validos += 1
                self.vitorias += 1
                resultado = "VITORIA"

            taxa_vitoria = (
                (self.vitorias / self.episodios_validos) * 100
                if self.episodios_validos > 0 else 0.0
            )

            linha = (
                f"{self._pc} | "
                f"Ep {self.episodio:4d} | "
                f"{resultado:8s} | "
                f"Passos: {info.get('passos', 0):6d} | "
                f"Tempo: {tempo_ep_minutos:7.2f} min | "
                f"Recompensa: {recompensa_ep:8.1f} | "
                f"Taxa vitória: {taxa_vitoria:.1f}%"
            )

            print(linha)
            self.arquivo_log.write(linha + "\n")

            ocorrido = info.get("ocorrido")
            if interrompido and ocorrido:
                linha_oc = (
                    f"{self._pc} | "
                    f"Ep {self.episodio:4d} | OCORRIDO | {ocorrido}"
                )
                print(linha_oc)
                self.arquivo_log.write(linha_oc + "\n")

            self.arquivo_log.flush()

        return True

    def _on_training_end(self):
        self.arquivo_log.write("Treino finalizado\n")
        self.arquivo_log.close()
        if self.arquivo_log_steps:
            self.arquivo_log_steps.write("Treino finalizado\n")
            self.arquivo_log_steps.close()


def treinar(
    timesteps: int = 500_000,
    carregar_modelo: str = None,
    carregar_ppo_antigo: str = None,
    log_steps: bool = False,
):
    """Treina com RecurrentPPO (LSTM).

    Args:
        timesteps: total de steps de treinamento.
        carregar_modelo: caminho para um RecurrentPPO salvo (.zip) para continuar.
        carregar_ppo_antigo: caminho para um PPO padrão salvo (.zip). Se fornecido,
            os pesos do extrator de features (CNN + MLP) são copiados para o novo
            RecurrentPPO. Útil para aproveitar o que o agente já aprendeu sobre
            a aparência do jogo sem precisar re-aprender do zero.
        log_steps: se True, imprime e salva o log de cada step.
    """
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    os.makedirs(PASTA_LOGS,    exist_ok=True)

    print("Iniciando ambiente FNAF1 com RecurrentPPO (LSTM)...")
    print("ATENÇÃO: Deixe o jogo aberto e na tela inicial!")
    print("Dica: segure F12 a qualquer momento para pausar.\n")
    time.sleep(3)

    # ── Cria ambiente ─────────────────────────────────────────────────────────
    env = DummyVecEnv([make_env_fn()])

    # ── Cria ou carrega o modelo ──────────────────────────────────────────────
    if carregar_modelo and os.path.exists(carregar_modelo):
        print(f"Carregando RecurrentPPO existente: {carregar_modelo}")
        modelo = RecurrentPPO.load(carregar_modelo, env=env)
    else:
        print("Criando novo modelo RecurrentPPO com LSTM...")
        policy_kwargs = dict(
            features_extractor_class=MultimodalExtractor,
            # LSTM: 256 unidades, 1 camada, compartilhada entre ator e crítico.
            # shared_lstm=True reduz parâmetros e é suficiente para este ambiente.
            # Para tarefas mais complexas ou futuras noites, considere shared_lstm=False.
            lstm_hidden_size=256,
            n_lstm_layers=1,
            shared_lstm=True,
            enable_critic_lstm=False,  # se shared_lstm=True, crítico usa o mesmo LSTM
        )
        modelo = RecurrentPPO(
            "MultiInputLstmPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            # n_steps=2048 com ~1000 steps/episódio → ~2 episódios por update.
            # Para gradientes mais estáveis, aumentar para 4096 se o treino
            # estiver muito ruidoso (ver docs/REFERENCIA_HIPERPARAMETROS.md).
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.995,
            # ent_coef=0.02: ligeiramente maior que o PPO original (0.01) para
            # combater a entropia colapsada após o plateau de -50 pontos.
            # Reduzir para 0.01 após o agente atingir taxa de vitória estável.
            ent_coef=0.02,
            verbose=0,
            tensorboard_log=PASTA_LOGS,
            device="auto",
        )

    # ── Transferência de pesos do PPO antigo ─────────────────────────────────
    if carregar_ppo_antigo and os.path.exists(carregar_ppo_antigo):
        print(f"\nTransferindo pesos do extrator de features: {carregar_ppo_antigo}")
        # DummyFNAFEnv: evita abrir o jogo apenas para carregar pesos
        env_temp = DummyVecEnv([DummyFNAFEnv])
        try:
            ppo_antigo = PPO.load(carregar_ppo_antigo, env=env_temp)
            modelo.policy.features_extractor.load_state_dict(
                ppo_antigo.policy.features_extractor.state_dict()
            )
            print("✓ Pesos transferidos com sucesso.")
            print("  O agente mantém o que aprendeu sobre a aparência do jogo.")
            print("  O LSTM ainda começará do zero — é necessário para a memória temporal.\n")
        except Exception as e:
            print(f"Aviso: falha na transferência de pesos — {e}")
            print("Continuando com pesos aleatórios no extrator.\n")
        finally:
            env_temp.close()
    elif carregar_ppo_antigo:
        print(f"Aviso: arquivo PPO antigo não encontrado: {carregar_ppo_antigo}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    pc = os.getenv("PC", "local")
    checkpoint = CheckpointCallback(
        save_freq=10_000,
        save_path=PASTA_MODELOS,
        name_prefix=f"{pc}_fnaf_recurrent_ppo",
    )
    log_callback = LogCallback(log_steps=log_steps)

    # ── Treinamento ───────────────────────────────────────────────────────────
    print(f"Treinando RecurrentPPO por {timesteps:,} timesteps...\n")
    try:
        modelo.learn(
            total_timesteps=timesteps,
            callback=[checkpoint, log_callback],
            reset_num_timesteps=carregar_modelo is None,
        )
    except KeyboardInterrupt:
        print("\nTreino interrompido pelo usuario. Salvando estado atual...")
    finally:
        if not log_callback.arquivo_log.closed:
            log_callback.arquivo_log.write("Treino finalizado\n")
            log_callback.arquivo_log.close()
        if log_callback.arquivo_log_steps and not log_callback.arquivo_log_steps.closed:
            log_callback.arquivo_log_steps.write("Treino finalizado\n")
            log_callback.arquivo_log_steps.close()

        caminho_final = f"{PASTA_MODELOS}/fnaf_recurrent_ppo_final"
        modelo.save(caminho_final)
        print(f"\nModelo final salvo em: {caminho_final}.zip")

        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Treino RecurrentPPO (LSTM) para FNAF")
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total de timesteps de treinamento (padrão: 500_000)",
    )
    parser.add_argument(
        "--modelo", type=str, default=None,
        help="Caminho para um RecurrentPPO salvo (.zip) para continuar treino",
    )
    parser.add_argument(
        "--ppo-antigo", type=str, default=None, dest="ppo_antigo",
        help=(
            "Caminho para um PPO padrão salvo (.zip). "
            "Transfere os pesos do extrator de features para o novo RecurrentPPO."
        ),
    )
    parser.add_argument(
        "--steps", action="store_true",
        help="Ativa log detalhado por step (verbose — impacto no desempenho)",
    )
    args = parser.parse_args()

    treinar(
        timesteps=args.timesteps,
        carregar_modelo=args.modelo,
        carregar_ppo_antigo=args.ppo_antigo,
        log_steps=args.steps,
    )
