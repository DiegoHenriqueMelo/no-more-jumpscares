import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from src.environment.fnaf_env import FNAFEnv
from src.agent.multimodal_policy import MultimodalExtractor
from pathlib import Path
from collections import Counter

NUM_ACOES = 17

class GameplayDataset(Dataset):
    def __init__(self, caminhos_json: list[str]):
        self.dados = []

        for caminho in caminhos_json:
            with open(caminho, "r") as f:
                dados = json.load(f)
                self.dados.extend(dados)
                print(f"Carregado: {caminho} ({len(dados)} frames)")

        print(f"\nTotal combinado: {len(self.dados)} frames")

        acoes_reais = [d for d in self.dados if d["acao"] != 0]
        print(f"Frames com ação real: {len(acoes_reais)}")

        contagem = Counter(d["nome"] for d in self.dados)
        print("\nDistribuição de ações:")
        for nome, qtd in contagem.most_common():
            print(f"  {nome}: {qtd}")

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, idx):
        dado = self.dados[idx]

        # Carrega imagem
        frame = cv2.imread(dado["frame"], cv2.IMREAD_GRAYSCALE)
        if frame is None:
            frame = np.zeros((84, 84), dtype=np.uint8)
        frame = np.expand_dims(frame, axis=-1)  # (84, 84, 1)

        # Estados: usa valores do JSON se existirem, senão usa zeros
        estados = np.array([
            float(dado.get("porta_esq", 0)),
            float(dado.get("porta_dir", 0)),
            float(dado.get("luz_esq", 0)),
            float(dado.get("luz_dir", 0)),
            float(dado.get("camera_aberta", 0)),
            float(dado.get("camera_ativa", 0)) / 11.0,
            float(dado.get("energia", 100)) / 100.0,
        ], dtype=np.float32)

        acao = int(dado["acao"])
        
        obs = {
            "imagem": torch.ByteTensor(frame),
            "estados": torch.FloatTensor(estados)
        }
        return obs, torch.LongTensor([acao])[0]


def treinar_bc(caminhos_json: list[str], epochs: int = 50, lr: float = 1e-3):
    print("=== Behavioral Cloning ===\n")

    dataset = GameplayDataset(caminhos_json)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    print("\nCriando modelo PPO com arquitetura multimodal...")
    env = FNAFEnv()
    
    policy_kwargs = dict(
        features_extractor_class=MultimodalExtractor,
    )
    
    modelo = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        verbose=0,
        device="auto",
    )

    policy    = modelo.policy
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    print(f"\nTreinando por {epochs} épocas...\n")

    for epoch in range(epochs):
        total_loss  = 0
        total_certo = 0
        total       = 0

        for obs_batch, acoes in loader:
            # Move observações para device
            obs_device = {
                "imagem": obs_batch["imagem"].to(modelo.device),
                "estados": obs_batch["estados"].to(modelo.device)
            }
            acoes = acoes.to(modelo.device)

            distribution = policy.get_distribution(obs_device)
            log_probs    = distribution.log_prob(acoes)
            loss         = -log_probs.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                acoes_pred = distribution.distribution.probs.argmax(dim=1)
                total_certo += (acoes_pred == acoes).sum().item()
                total       += len(acoes)

        acuracia = total_certo / total * 100
        print(f"Época {epoch+1:3d}/{epochs} | Loss: {total_loss/len(loader):.4f} | Acurácia: {acuracia:.1f}%")

    caminho_saida = "modelos/fnaf_bc.zip"
    modelo.save(caminho_saida)
    print(f"\nModelo BC salvo em: {caminho_saida}")

    env.close()
    return modelo


def combinar_bc_com_ppo(caminho_bc: str = "modelos/fnaf_bc.zip",
                         caminho_ppo: str = "modelos/fnaf_merged.zip"):
    print("Combinando BC + PPO...")
    env = FNAFEnv()

    modelo_bc  = PPO.load(caminho_bc,  env=env)
    modelo_ppo = PPO.load(caminho_ppo, env=env)

    params_bc  = modelo_bc.policy.state_dict()
    params_ppo = modelo_ppo.policy.state_dict()

    params_combined = {}
    for chave in params_bc:
        params_combined[chave] = 0.3 * params_bc[chave] + 0.7 * params_ppo[chave]

    modelo_ppo.policy.load_state_dict(params_combined)

    caminho_saida = "modelos/fnaf_bc_ppo.zip"
    modelo_ppo.save(caminho_saida)
    print(f"Modelo combinado salvo em: {caminho_saida}")

    env.close()


if __name__ == "__main__":
    treinar_bc([
        "dados/gameplay_teste/dataset.json",
    ], epochs=200)

    combinar_bc_com_ppo(
        caminho_bc  = "modelos/fnaf_bc.zip",
        caminho_ppo = "modelos/fnaf_merged.zip"
    )

    print("\nPronto! Use modelos/fnaf_bc_ppo.zip como base para o próximo treino.")