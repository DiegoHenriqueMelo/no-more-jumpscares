"""Behavioral Cloning (BC) para FNAF — warmup antes do PPO/RecurrentPPO.

Fluxo recomendado
-----------------
1. Grave gameplay humano:
       python -m src.utils.gravar_gameplay

2. Treine o BC:
       python -m src.agent.behavioral_cloning --dados dados/gameplay_*/dataset.json

3. Use o modelo BC como ponto de partida para RecurrentPPO:
       python -m src.agent.train_recurrent --ppo-antigo modelos/fnaf_bc.zip

   (O train_recurrent transfere os pesos do extrator de features do BC para o
   novo RecurrentPPO com LSTM. O LSTM começa zerado — isso é esperado.)

Compatibilidade com RecurrentPPO
----------------------------------
O BC treina um PPO padrão (sem LSTM). Isso é intencional: o BC aprende o mapeamento
frame → ação sobre exemplos independentes, onde não há sequência temporal. O LSTM
do RecurrentPPO é adicionado depois, durante o treino com RL.

O extrator de features (CNN + MLP) é arquiteturalmente idêntico entre PPO e
RecurrentPPO — o que permite a transferência de pesos via --ppo-antigo.

Estados no dataset
-------------------
O vetor de estados tem 9 dimensões (mesma ordem que o observation_space):
    [porta_esq, porta_dir, luz_esq, luz_dir, camera_aberta,
     camera_ativa/11.0, energia/100.0, tempo_ep/535.0, cooldown_camera>0]

O script gravar_gameplay.py registra todos os 9 campos. Datasets antigos
(com 7 ou 8 estados) são automaticamente completados com zeros nas dimensões
ausentes — subótimo mas não quebra o treinamento.
"""
import json
import sys
import numpy as np
import cv2
import torch
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from src.agent.multimodal_policy import MultimodalExtractor
from src.utils.dummy_env import DummyFNAFEnv

NUM_ACOES = 17
ALTURA    = 84
LARGURA   = 84


# ── Dataset ───────────────────────────────────────────────────────────────────

class GameplayDataset(Dataset):
    """Dataset de gameplay humano para BC.

    Carrega um ou mais arquivos JSON gerados por gravar_gameplay.py.

    Formato de cada entrada no JSON:
        {
            "frame": "dados/.../frames/000000.png",
            "acao": 5,
            "nome": "abrir_fechar_camera",
            "porta_esq": 0,
            "porta_dir": 1,
            "luz_esq": 0,
            "luz_dir": 0,
            "camera_aberta": 0,
            "camera_ativa": 0,
            "energia": 87.3,
            "tempo_ep": 12.5
        }

    Datasets antigos sem os campos de estado são aceitos: os campos ausentes
    recebem valor zero. O campo "tempo_ep" (8ª dimensão) é o mais comumente
    ausente em datasets antigos.
    """

    def __init__(self, caminhos_json: list[str], max_nada_ratio: float = 2.0):
        """
        Args:
            caminhos_json: lista de arquivos dataset.json.
            max_nada_ratio: máximo de frames "nada" por frame de ação real.
                2.0 (padrão) → no máximo 2 "nada" para cada 1 ação.
                0.0          → remove todos os "nada".
                float("inf") → mantém todos (comportamento antigo).
                Ex: se há 500 ações reais e 4000 "nada", com ratio=2.0
                apenas 1000 "nada" são mantidos, descartando 3000.
        """
        brutos = []

        for caminho in caminhos_json:
            with open(caminho, "r") as f:
                dados = json.load(f)
                brutos.extend(dados)
                print(f"Carregado: {caminho} ({len(dados)} frames)")

        acoes_reais = [d for d in brutos if d["acao"] != 0]
        frames_nada = [d for d in brutos if d["acao"] == 0]

        # Limita "nada" para evitar que o modelo aprenda a ficar parado
        limite_nada = int(len(acoes_reais) * max_nada_ratio)
        if len(frames_nada) > limite_nada:
            # Distribui o corte uniformemente ao longo do tempo (não apenas
            # descarta os últimos) para preservar a distribuição temporal.
            passo = len(frames_nada) / limite_nada
            frames_nada = [frames_nada[int(i * passo)] for i in range(limite_nada)]
            print(
                f"\nFrames 'nada' reduzidos: {len([d for d in brutos if d['acao'] == 0])} "
                f"→ {len(frames_nada)} (ratio {max_nada_ratio}x ações reais)"
            )

        self.dados = acoes_reais + frames_nada
        # Re-embaralha para misturar "nada" e ações ao longo das épocas
        import random
        random.shuffle(self.dados)

        print(f"\nTotal após balanceamento: {len(self.dados)} frames")
        print(f"  Ações reais : {len(acoes_reais)}")
        print(f"  'nada'      : {len(frames_nada)}")

        contagem = Counter(d["nome"] for d in self.dados)
        print("\nDistribuição de ações:")
        for nome, qtd in contagem.most_common():
            print(f"  {nome}: {qtd}")

        # Avisa sobre campos ausentes em datasets antigos
        tem_tempo = any("tempo_ep" in d for d in self.dados[:10])
        tem_cooldown = any("cooldown_camera" in d for d in self.dados[:10])
        if not tem_tempo:
            print(
                "\nAviso: dataset sem 'tempo_ep' (8ª dimensão). "
                "Dimensão será zero — subótimo mas funcional."
            )
        if not tem_cooldown:
            print(
                "\nAviso: dataset sem 'cooldown_camera' (9ª dimensão). "
                "Dimensão será zero — subótimo mas funcional."
            )

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, idx):
        dado = self.dados[idx]

        # Carrega e processa imagem
        frame = cv2.imread(dado["frame"], cv2.IMREAD_GRAYSCALE)
        if frame is None:
            frame = np.zeros((ALTURA, LARGURA), dtype=np.uint8)
        frame = cv2.resize(frame, (LARGURA, ALTURA))
        frame = np.expand_dims(frame, axis=-1)  # (84, 84, 1)

        # Estados: 9 dimensões (mesma ordem que FNAFEnv.observation_space)
        estados = np.array([
            float(dado.get("porta_esq",       0)),
            float(dado.get("porta_dir",       0)),
            float(dado.get("luz_esq",         0)),
            float(dado.get("luz_dir",         0)),
            float(dado.get("camera_aberta",   0)),
            float(dado.get("camera_ativa",    0)) / 11.0,
            float(dado.get("energia",         100)) / 100.0,
            float(dado.get("tempo_ep",        0)) / 535.0,
            float(dado.get("cooldown_camera", 0)),            # 9ª dimensão
        ], dtype=np.float32)

        acao = int(dado["acao"])

        obs = {
            "imagem":  torch.ByteTensor(frame),
            "estados": torch.FloatTensor(estados),
        }
        return obs, torch.LongTensor([acao])[0]


# ── Treinamento BC ────────────────────────────────────────────────────────────

def treinar_bc(
    caminhos_json: list[str],
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    max_nada_ratio: float = 2.0,
) -> PPO:
    """Treina um modelo PPO via Behavioral Cloning.

    O modelo resultante pode ser usado como warmup para RecurrentPPO via
    train_recurrent.py --ppo-antigo modelos/fnaf_bc.zip

    Args:
        caminhos_json: lista de caminhos para arquivos JSON de gameplay.
        epochs: número de épocas de treinamento.
        lr: learning rate do otimizador Adam.
        batch_size: tamanho do batch.
        max_nada_ratio: máximo de frames "nada" por frame de ação real (padrão: 2.0).
            Evita que o modelo aprenda a ficar parado porque "nada" domina o dataset.

    Returns:
        Modelo PPO treinado (sem LSTM — use --ppo-antigo para combinar com RecurrentPPO).
    """
    print("=== Behavioral Cloning ===\n")

    dataset = GameplayDataset(caminhos_json, max_nada_ratio=max_nada_ratio)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Cria PPO com ambiente dummy — sem abrir o jogo
    print("\nCriando modelo PPO (sem LSTM) para BC...")
    env_dummy = DummyVecEnv([DummyFNAFEnv])

    policy_kwargs = dict(
        features_extractor_class=MultimodalExtractor,
    )

    modelo = PPO(
        policy="MultiInputPolicy",
        env=env_dummy,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        verbose=0,
        device="auto",
    )

    policy    = modelo.policy
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Pesos de classe: cada ação recebe peso = total / (n_classes × contagem).
    # Isso impede que ações raras (ex: camera_7) sejam ignoradas pelo modelo
    # em favor de "nada", que domina o dataset mesmo após o balanceamento.
    contagem_acoes = Counter(d["acao"] for d in dataset.dados)
    pesos_classe = torch.ones(NUM_ACOES, device=modelo.device)
    total_amostras = len(dataset.dados)
    for acao_id, count in contagem_acoes.items():
        pesos_classe[acao_id] = total_amostras / (NUM_ACOES * count)
    print("Pesos de classe (inverso da frequência):")
    for acao_id in sorted(contagem_acoes):
        from src.environment.fnaf_env import ACOES
        print(f"  {ACOES[acao_id]:25s}: {pesos_classe[acao_id].item():.3f}")

    print(f"\nTreinando por {epochs} épocas (batch={batch_size}, lr={lr})...\n")

    melhor_acuracia = 0.0

    for epoch in range(epochs):
        total_loss  = 0.0
        total_certo = 0
        total       = 0

        for obs_batch, acoes in loader:
            obs_device = {
                "imagem":  obs_batch["imagem"].to(modelo.device),
                "estados": obs_batch["estados"].to(modelo.device),
            }
            acoes = acoes.to(modelo.device)

            distribution = policy.get_distribution(obs_device)
            log_probs    = distribution.log_prob(acoes)

            # Pondera o loss: ações raras têm peso maior, "nada" tem peso menor
            pesos_batch = pesos_classe[acoes]
            loss        = -(log_probs * pesos_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                acoes_pred = distribution.distribution.probs.argmax(dim=1)
                total_certo += (acoes_pred == acoes).sum().item()
                total       += len(acoes)

        acuracia = total_certo / total * 100
        melhor_acuracia = max(melhor_acuracia, acuracia)
        print(
            f"Época {epoch+1:3d}/{epochs} | "
            f"Loss: {total_loss/len(loader):.4f} | "
            f"Acurácia: {acuracia:.1f}%"
        )

    print(f"\nMelhor acurácia atingida: {melhor_acuracia:.1f}%")

    caminho_saida = "modelos/fnaf_bc.zip"
    modelo.save(caminho_saida)
    print(f"Modelo BC salvo em: {caminho_saida}")
    print(
        "\nPróximo passo:\n"
        "  python -m src.agent.train_recurrent --ppo-antigo modelos/fnaf_bc.zip\n"
        "Isso transfere os pesos do extrator de features do BC para o RecurrentPPO."
    )

    env_dummy.close()
    return modelo


# ── Utilitário: transferência para RecurrentPPO ───────────────────────────────

def transferir_para_recurrent(
    caminho_bc: str = "modelos/fnaf_bc.zip",
    caminho_recurrent: str = "modelos/fnaf_recurrent_ppo_final.zip",
    caminho_saida: str = "modelos/fnaf_recurrent_bc_init.zip",
):
    """Copia os pesos do extrator de features do BC para um RecurrentPPO treinado.

    Use este utilitário se quiser inicializar um RecurrentPPO com pesos BC
    sem passar pelo train_recurrent.py (ex: experimento, comparação).

    Args:
        caminho_bc: modelo PPO treinado com BC.
        caminho_recurrent: modelo RecurrentPPO a receber os pesos.
        caminho_saida: onde salvar o RecurrentPPO com pesos BC.
    """
    print("Transferindo pesos BC → RecurrentPPO...")
    env_dummy = DummyVecEnv([DummyFNAFEnv])

    modelo_bc        = PPO.load(caminho_bc, env=env_dummy)
    modelo_recurrent = RecurrentPPO.load(caminho_recurrent, env=env_dummy)

    modelo_recurrent.policy.features_extractor.load_state_dict(
        modelo_bc.policy.features_extractor.state_dict()
    )

    modelo_recurrent.save(caminho_saida)
    print(f"Salvo em: {caminho_saida}")

    env_dummy.close()


# ── Utilitário legado: combinar BC + PPO padrão ───────────────────────────────

def combinar_bc_com_ppo(
    caminho_bc:  str = "modelos/fnaf_bc.zip",
    caminho_ppo: str = "modelos/fnaf_ppo_final.zip",
    peso_bc:     float = 0.3,
):
    """Mescla pesos de um modelo BC com um PPO treinado (interpolação linear).

    Produz um modelo PPO combinado. Para integrar com RecurrentPPO, use depois:
        python -m src.agent.train_recurrent --ppo-antigo modelos/fnaf_bc_ppo.zip

    Args:
        caminho_bc: modelo PPO treinado com BC.
        caminho_ppo: modelo PPO treinado com RL.
        peso_bc: peso do BC na interpolação (padrão: 0.3 = 30% BC + 70% RL).
    """
    print(f"Combinando BC ({peso_bc*100:.0f}%) + PPO ({(1-peso_bc)*100:.0f}%)...")
    env_dummy = DummyVecEnv([DummyFNAFEnv])

    modelo_bc  = PPO.load(caminho_bc,  env=env_dummy)
    modelo_ppo = PPO.load(caminho_ppo, env=env_dummy)

    params_bc  = modelo_bc.policy.state_dict()
    params_ppo = modelo_ppo.policy.state_dict()

    params_combined = {
        k: peso_bc * params_bc[k] + (1 - peso_bc) * params_ppo[k]
        for k in params_bc
    }

    modelo_ppo.policy.load_state_dict(params_combined)

    caminho_saida = "modelos/fnaf_bc_ppo.zip"
    modelo_ppo.save(caminho_saida)
    print(f"Modelo combinado salvo em: {caminho_saida}")

    env_dummy.close()


# ── Ponto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Behavioral Cloning para FNAF")
    parser.add_argument(
        "--dados", nargs="+", required=True,
        help=(
            "Caminhos para arquivos JSON de gameplay. "
            "Aceita glob: --dados dados/gameplay_*/dataset.json"
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Número de épocas de treinamento (padrão: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (padrão: 1e-3)",
    )
    parser.add_argument(
        "--batch", type=int, default=32,
        help="Tamanho do batch (padrão: 32)",
    )
    parser.add_argument(
        "--nada-ratio", type=float, default=2.0,
        dest="nada_ratio",
        help=(
            "Máximo de frames 'nada' por frame de ação real (padrão: 2.0). "
            "Use 0 para remover todos os 'nada', inf para manter todos."
        ),
    )
    args = parser.parse_args()

    # Expande globs
    caminhos = []
    for padrao in args.dados:
        expandidos = glob.glob(padrao)
        if expandidos:
            caminhos.extend(expandidos)
        else:
            caminhos.append(padrao)  # mantém como está (pode ser caminho direto)

    if not caminhos:
        print("Erro: nenhum arquivo JSON encontrado com os padrões fornecidos.")
        sys.exit(1)

    treinar_bc(
        caminhos_json=caminhos,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        max_nada_ratio=args.nada_ratio,
    )
