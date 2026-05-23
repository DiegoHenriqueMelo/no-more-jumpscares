# Como usar o projeto — guia completo

Este guia descreve todos os scripts do projeto, o que cada um faz, e como executar
cada fluxo de ponta a ponta.

---

## Pré-requisitos

1. **Python 3.11+** com o ambiente virtual ativado
2. **Dependências instaladas:**
   ```
   pip install -r requirements.txt
   ```
3. **`.env` configurado** com as variáveis do seu PC (copie `.env.example`):
   ```
   FNAF_WINDOW_TITLE=Five Nights at Freddy's
   PC=pc1
   FNAF_RESET_CLICK_X=960
   FNAF_RESET_CLICK_Y=540
   # ... (demais coordenadas das ações)
   ```
4. **Imagens de referência calibradas** (necessárias para detectar morte/vitória):
   ```
   python -m src.utils.calibrar morte
   python -m src.utils.calibrar vitoria
   python -m src.utils.calibrar camera_aberta   # opcional, melhora detecção
   ```
5. **O jogo FNAF1 aberto** e posicionado na tela conforme as coordenadas do `.env`

---

## Fluxo recomendado

```
[Opcional] Gravar gameplay humano
         ↓
[Opcional] Treinar Behavioral Cloning
         ↓
Treinar RecurrentPPO (LSTM)          ← fluxo principal
         ↓
[Multi-PC] Mesclar modelos de vários PCs
```

---

## Scripts disponíveis

### 1. Treino com RecurrentPPO (recomendado)

**Arquivo:** `src/agent/train_recurrent.py`  
**Algoritmo:** RecurrentPPO com LSTM — o agente tem memória entre frames.

#### Novo treinamento (do zero)

```bash
python -m src.agent.train_recurrent
# Padrão: 500.000 timesteps
```

#### Especificar número de timesteps

```bash
python -m src.agent.train_recurrent --timesteps 1000000
```

#### Continuar de um checkpoint

```bash
python -m src.agent.train_recurrent --modelo modelos/fnaf_recurrent_ppo_final.zip
```

O modelo final é salvo em `modelos/fnaf_recurrent_ppo_final.zip`.
Checkpoints intermediários são salvos a cada 10.000 steps em `modelos/`.

#### Transferir pesos de um PPO antigo

Se você já tem um modelo PPO treinado (ou um modelo BC), pode aproveitar os pesos
do extrator de features (CNN + MLP):

```bash
# A partir de um PPO padrão:
python -m src.agent.train_recurrent --ppo-antigo modelos/fnaf_ppo_final.zip

# A partir de um modelo BC:
python -m src.agent.train_recurrent --ppo-antigo modelos/fnaf_bc.zip
```

Isso preserva o que o agente aprendeu sobre a **aparência** do jogo (reconhecimento
visual). O LSTM começa do zero — é esperado, pois a memória temporal é nova.

#### Log detalhado por step

```bash
python -m src.agent.train_recurrent --steps
```

Imprime o estado completo a cada step: energia, portas, luzes, câmera, ação
executada. Útil para diagnóstico, mas tem impacto no desempenho.

#### Múltiplas janelas do jogo (VecEnv)

Para N janelas abertas do FNAF simultaneamente:

```bash
python -m src.agent.train_recurrent --n-envs 2
```

> **Atenção:** cada instância precisa de sua própria janela do jogo com título
> diferente. Configure `window_title_override` e `coord_offset` no código se as
> janelas estiverem em posições diferentes na tela.

#### Pausar durante o treino

Segure **F12** a qualquer momento para pausar. Solte para continuar.

---

### 2. Treino com PPO padrão (legado)

**Arquivo:** `src/agent/train.py`  
O PPO padrão sem LSTM. Mantido para compatibilidade com modelos antigos.
Para novos treinos, prefira o `train_recurrent.py`.

```bash
python -m src.agent.train
python -m src.agent.train --timesteps 500000
python -m src.agent.train --modelo modelos/fnaf_ppo_final.zip
```

---

### 3. Gravar gameplay humano

**Arquivo:** `src/utils/gravar_gameplay.py`  
Grava a sua gameplay para criar um dataset de Behavioral Cloning.

```bash
python -m src.utils.gravar_gameplay
```

O script aguarda 5 segundos para você focar o jogo, depois começa a gravar.

**Controles durante a gravação:**

| Tecla | Ação |
|-------|------|
| A | porta_esquerda (toggle) |
| D | porta_direita (toggle) |
| Q | luz_esquerda (toggle) |
| E | luz_direita (toggle) |
| Tab | abrir/fechar câmera |
| 1–9, 0, - | câmeras (1a, 1b, 1c, 2a, 2b, 3, 4a, 4b, 5, 6, 7) |
| **F10** | **para a gravação e salva o dataset** |

Todos os 8 estados são registrados automaticamente (energia estimada, tempo, etc.).

O dataset é salvo em `dados/gameplay_YYYYMMDD_HHMMSS/dataset.json`.

#### Gravar múltiplas partidas

Execute o script várias vezes. Para treinar o BC com todos os datasets:

```bash
python -m src.agent.behavioral_cloning --dados dados/gameplay_*/dataset.json
```

---

### 4. Behavioral Cloning (warmup supervisionado)

**Arquivo:** `src/agent/behavioral_cloning.py`  
Treina um modelo por imitação do gameplay humano antes do RL. O agente aprende
as mecânicas básicas sem precisar rodando o jogo.

```bash
# Com um dataset:
python -m src.agent.behavioral_cloning --dados dados/gameplay_20240520_143000/dataset.json

# Com múltiplos datasets (glob):
python -m src.agent.behavioral_cloning --dados dados/gameplay_*/dataset.json

# Ajustar hiperparâmetros:
python -m src.agent.behavioral_cloning --dados dados/gameplay_*/dataset.json --epochs 200 --lr 1e-4 --batch 64
```

**Parâmetros:**

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--dados` | (obrigatório) | Caminhos para os arquivos `dataset.json` |
| `--epochs` | 100 | Épocas de treinamento |
| `--lr` | 1e-3 | Learning rate |
| `--batch` | 32 | Tamanho do batch |

O modelo BC é salvo em `modelos/fnaf_bc.zip`.

**Após o BC, inicie o RecurrentPPO com os pesos pré-treinados:**

```bash
python -m src.agent.train_recurrent --ppo-antigo modelos/fnaf_bc.zip
```

---

### 5. Mesclar modelos de vários PCs

**Arquivo:** `merge_modelos.py`  
Combina os pesos de múltiplos modelos PPO treinados em máquinas diferentes
(equivalente a Federated Learning simplificado).

```bash
# Mesclar 2 modelos:
python merge_modelos.py modelos/pc1_fnaf.zip modelos/pc2_fnaf.zip

# Mesclar 3 ou mais:
python merge_modelos.py modelos/pc1.zip modelos/pc2.zip modelos/pc3.zip
```

O modelo mesclado é salvo em `modelos/fnaf_merged.zip`.

Para usar o modelo mesclado como ponto de partida para RecurrentPPO:

```bash
python -m src.agent.train_recurrent --ppo-antigo modelos/fnaf_merged.zip
```

---

### 6. Calibração das referências

**Arquivo:** `src/utils/calibrar.py`  
Captura screenshots do jogo para criar os templates de detecção.

```bash
# Capturar referência de morte (tela "Game Over"):
python -m src.utils.calibrar morte

# Capturar referência de vitória (tela "6 AM"):
python -m src.utils.calibrar vitoria

# Capturar referência de câmera aberta (indicador "YOU" no mapa):
python -m src.utils.calibrar camera_aberta
```

Execute cada comando quando a tela correspondente estiver visível no jogo.
As imagens são salvas em `src/utils/referencias/`.

---

### 7. Testar detecção de morte/vitória

**Arquivo:** `src/utils/testar_deteccao.py`  
Verifica se os templates de detecção estão funcionando.

```bash
python -m src.utils.testar_deteccao
```

---

### 8. Simular modelo de energia

**Arquivo:** `src/utils/simular_energia.py`  
Simula a curva de energia com diferentes estratégias para verificar se o modelo
corresponde ao jogo real.

```bash
python -m src.utils.simular_energia
```

---

## Logs gerados

| Arquivo | Conteúdo |
|---------|---------|
| `logs/treino_recurrent.log` | Resultado por episódio (RecurrentPPO) |
| `logs/treino_recurrent_steps.log` | Log por step (apenas com `--steps`) |
| `logs/treino.log` | Resultado por episódio (PPO padrão) |
| `logs/desyncs.log` | Contagem de correções de estado por episódio |
| `logs/` (tensorboard) | Métricas de treinamento para TensorBoard |

Para visualizar os logs no TensorBoard:

```bash
tensorboard --logdir logs/
```

---

## Modelos salvos

| Arquivo | Descrição |
|---------|-----------|
| `modelos/fnaf_recurrent_ppo_final.zip` | Modelo RecurrentPPO ao final do treino |
| `modelos/{PC}_fnaf_recurrent_ppo_{N}_steps.zip` | Checkpoint a cada 10.000 steps |
| `modelos/fnaf_bc.zip` | Modelo após Behavioral Cloning |
| `modelos/fnaf_merged.zip` | Modelo mesclado de vários PCs |
| `modelos/fnaf_ppo_final.zip` | Modelo PPO padrão (legado) |

---

## Diagnóstico — o que fazer quando o agente empaca

### Plateau de recompensa (ex: −50 a −100 por 500+ episódios)

1. **Verifique o `ent_coef`** — se a política ficou determinística, aumente para 0.03–0.05
2. **Verifique os logs por step** (`--steps`) — qual componente de recompensa domina?
3. **RecurrentPPO** resolve o plateau causado por falta de memória temporal (o agente
   não conseguia rastrear posições de animatrônicos entre frames)
4. **Behavioral Cloning** — grave uma gameplay de boa qualidade e use como warmup

### Agente nunca usa câmeras

Verifique `logs/desyncs.log` — se `SYNC camera` é sempre 0, o agente está evitando
câmeras. Isso geralmente indica penalidade de repetição mal calibrada ou
`ent_coef` muito baixo.

### Jogo não é detectado

Verifique se `FNAF_WINDOW_TITLE` no `.env` corresponde exatamente ao título da
janela do jogo (case-sensitive em alguns sistemas).

### Detecção de morte/vitória incorreta

Execute a calibração novamente com o jogo nas condições corretas (resolução, modo
janela, posição na tela). As referências são específicas por PC.
