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

VecEnv (múltiplas janelas)
---------------------------
Para rodar N instâncias paralelas do FNAF, use a factory make_env_fn() e passe
uma lista de configurações (window_title_override + coord_offset por janela).
O RecurrentPPO suporta VecEnv nativamente — os estados LSTM são rastreados por
sub-ambiente internamente pelo sb3-contrib.

    envs = SubprocVecEnv([
        make_env_fn("FNAF Janela 1", coord_offset=(0, 0)),
        make_env_fn("FNAF Janela 2", coord_offset=(1280, 0)),
    ])

Requer: múltiplas janelas do jogo abertas simultaneamente, cada uma com título
distinto configurado no .env (FNAF_WINDOW_TITLE_1, FNAF_WINDOW_TITLE_2, etc.)
e coordenadas ajustadas via coord_offset.

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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.environment.fnaf_env import FNAFEnv
from src.agent.multimodal_policy import MultimodalExtractor
from src.utils.dummy_env import DummyFNAFEnv

PASTA_MODELOS = "modelos"
PASTA_LOGS    = "logs"


def make_env_fn(window_title_override: str = None, coord_offset: tuple = (0, 0)):
    """Retorna uma função que cria um FNAFEnv com configuração específica de janela.

    Projetado para uso com DummyVecEnv ou SubprocVecEnv:

        envs = SubprocVecEnv([
            make_env_fn("FNAF - Janela 1", coord_offset=(0, 0)),
            make_env_fn("FNAF - Janela 2", coord_offset=(1280, 0)),
        ])

    Args:
        window_title_override: título da janela do jogo a controlar. Se None,
            usa FNAF_WINDOW_TITLE do .env.
        coord_offset: (dx, dy) somado a todas as coordenadas de clique. Use
            para múltiplas janelas posicionadas em grid na tela.
    """
    def _init():
        return FNAFEnv(
            window_title_override=window_title_override,
            coord_offset=coord_offset,
        )
    return _init


class LogCallback(BaseCallback):
    """Loga progresso do treinamento step a step e por episódio.

    Diferenças em relação ao train.py original:
    - Suporta VecEnv com n_envs > 1 (itera sobre todos os ambientes no done).
    - Mantém recompensa acumulada por sub-ambiente para log correto no VecEnv.
    """

    def __init__(self, log_steps: bool = False):
        super().__init__()
        self.episodio          = 0
        self.episodios_validos = 0
        self.mortes            = 0
        self.vitorias          = 0
        self.interrompidos     = 0
        self._recompensas      = {}  # env_idx → recompensa acumulada no episódio
        self._pausa_disponivel = True
        self._log_steps        = log_steps
        self._pc               = os.getenv("PC", "")
        self.n_envs            = 1  # atualizado em _on_training_start

        os.makedirs("logs", exist_ok=True)
        cabecalho = f"\n{'='*60}\nTreino RecurrentPPO (LSTM) iniciado\n{'='*60}\n"

        self.arquivo_log = open("logs/treino_recurrent.log", "a", encoding="utf-8")
        self.arquivo_log.write(cabecalho)

        self.arquivo_log_steps = None
        if log_steps:
            self.arquivo_log_steps = open("logs/treino_recurrent_steps.log", "a", encoding="utf-8")
            self.arquivo_log_steps.write(cabecalho)

    def _on_training_start(self):
        # Captura n_envs do modelo para label correto no log de episódio
        self.n_envs = self.training_env.num_envs

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

        infos   = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])
        dones   = self.locals.get("dones", [False])

        for env_idx, (info, reward, done) in enumerate(zip(infos, rewards, dones)):
            # Acumula recompensa por ambiente
            self._recompensas[env_idx] = self._recompensas.get(env_idx, 0.0) + reward

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
                    f"Env {env_idx} | "
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
                recompensa_ep    = self._recompensas.pop(env_idx, 0.0)

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

                env_label = f" Env{env_idx}" if self.n_envs > 1 else ""
                linha = (
                    f"{self._pc}{env_label} | "
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
    n_envs: int = 1,
    env_configs: list = None,
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
        n_envs: número de ambientes paralelos. Para n_envs > 1, forneça env_configs.
        env_configs: lista de dicts com kwargs para make_env_fn() por ambiente.
            Exemplo: [
                {"window_title_override": "FNAF 1", "coord_offset": (0, 0)},
                {"window_title_override": "FNAF 2", "coord_offset": (1280, 0)},
            ]
        log_steps: se True, imprime e salva o log de cada step.
    """
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    os.makedirs(PASTA_LOGS,    exist_ok=True)

    print("Iniciando ambiente FNAF1 com RecurrentPPO (LSTM)...")
    print("ATENÇÃO: Deixe o jogo aberto e na tela inicial!")
    print("Dica: segure F12 a qualquer momento para pausar.\n")
    time.sleep(3)

    # ── Cria ambientes ────────────────────────────────────────────────────────
    if n_envs == 1:
        env = DummyVecEnv([make_env_fn()])
        print("Modo: ambiente único (DummyVecEnv)")
    else:
        if env_configs and len(env_configs) == n_envs:
            env_fns = [make_env_fn(**cfg) for cfg in env_configs]
        else:
            print(
                f"Aviso: env_configs não fornecido ou tamanho incorreto para "
                f"n_envs={n_envs}. Usando n_envs janelas sem override."
            )
            env_fns = [make_env_fn() for _ in range(n_envs)]
        env = SubprocVecEnv(env_fns)
        print(f"Modo: {n_envs} ambientes paralelos (SubprocVecEnv)")

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
    parser.add_argument(
        "--n-envs", type=int, default=1, dest="n_envs",
        help="Número de ambientes paralelos (padrão: 1). Requer múltiplas janelas do jogo.",
    )
    args = parser.parse_args()

    treinar(
        timesteps=args.timesteps,
        carregar_modelo=args.modelo,
        carregar_ppo_antigo=args.ppo_antigo,
        n_envs=args.n_envs,
        log_steps=args.steps,
    )
