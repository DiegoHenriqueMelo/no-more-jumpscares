# Documentação Completa das Alterações - Sessão de Melhorias

## Resumo Executivo

Esta sessão implementou melhorias fundamentais no ambiente de aprendizado por reforço para FNAF 1, focando em:
1. **Observabilidade aumentada** - Estados internos explícitos
2. **Mecânicas realistas** - Sistema de energia e tempo baseado no jogo real
3. **Validação de ações** - Contexto determina ações válidas
4. **Sistema de recompensas equilibrado** - Incentivos e penalidades balanceados

---

## 1. ESTADOS INTERNOS (Observabilidade)

### 1.1 Mudança no Observation Space

**ANTES:**
```python
observation_space = spaces.Box(
    low=0, high=255,
    shape=(84, 84, 1),  # Apenas imagem
    dtype=np.uint8
)
```

**DEPOIS:**
```python
observation_space = spaces.Dict({
    "imagem": spaces.Box(
        low=0, high=255,
        shape=(84, 84, 1),
        dtype=np.uint8
    ),
    "estados": spaces.Box(
        low=0, high=1,
        shape=(7,),  # 7 estados normalizados
        dtype=np.float32
    )
})
```

### 1.2 Vetor de Estados (7 dimensões)

| Índice | Estado | Valores | Descrição |
|--------|--------|---------|-----------|
| 0 | `porta_esq` | 0.0 ou 1.0 | Porta esquerda fechada |
| 1 | `porta_dir` | 0.0 ou 1.0 | Porta direita fechada |
| 2 | `luz_esq` | 0.0 ou 1.0 | Luz esquerda ligada |
| 3 | `luz_dir` | 0.0 ou 1.0 | Luz direita ligada |
| 4 | `camera_aberta` | 0.0 ou 1.0 | Sistema de câmera aberto |
| 5 | `camera_ativa` | 0.0 a 1.0 | Qual câmera (normalizado /11) |
| 6 | `energia` | 0.0 a 1.0 | Energia restante (normalizado /100) |

### 1.3 Rastreamento de Estados

**Novos atributos adicionados:**
```python
self.luz_esq = False
self.luz_dir = False
self.luz_esq_timer = 0  # Auto-desliga após 1 step
self.luz_dir_timer = 0
self.tempo_jogo = 0.0
self.penultima_acao = None  # Para detectar spam
self.ultimo_update_energia = time.perf_counter()
```

---

## 2. SISTEMA DE ENERGIA (Baseado no FNAF 1 Real)

### 2.1 Fórmula de Consumo

**Implementação:**
```python
def _atualizar_energia(self):
    dt = agora - self.ultimo_update_energia
    
    usage = 1  # Base (ventilador)
    usage += int(self.porta_esq)
    usage += int(self.porta_dir)
    usage += int(self.luz_esq)
    usage += int(self.luz_dir)
    usage += int(self.camera_aberta)
    usage = min(usage, 4)  # Máximo 4 barras
    
    consumo_por_segundo = usage * 0.1  # 0.1% por barra/segundo
    self.energia -= consumo_por_segundo * dt
    self.energia = max(0.0, self.energia)
```

### 2.2 Tabela de Consumo

| Sistemas Ativos | Usage | Consumo/s | Duração Total |
|-----------------|-------|-----------|---------------|
| Nenhum (só ventilador) | 1 | 0.1%/s | ~1000s (16min) |
| 1 sistema | 2 | 0.2%/s | ~500s (8min) |
| 2 sistemas | 3 | 0.3%/s | ~333s (5.5min) |
| 3+ sistemas | 4 | 0.4%/s | ~250s (4min) |

**Nota:** Noite completa = 535s (~9min). Com 4 barras, energia acaba antes do 6 AM.

### 2.3 Comportamento quando Energia Acaba

**ANTES:**
```python
terminado = self.energia <= 0  # Termina imediatamente
```

**DEPOIS:**
```python
if self.energia <= 0:
    # Desliga todos os sistemas (realista)
    self.porta_esq = False
    self.porta_dir = False
    self.luz_esq = False
    self.luz_dir = False
    self.camera_aberta = False
    
    # Aguarda detecção de morte (Freddy demora alguns segundos)
    time.sleep(STEP_DELAY)
    morreu = self._detectar_morte()
    
    return observacao, -500.0, True, False, info
```

---

## 3. SISTEMA DE TEMPO

### 3.1 Implementação

```python
def _atualizar_tempo(self):
    self.tempo_jogo += STEP_DELAY  # Acumula tempo linear
```

### 3.2 Duração das Horas (FNAF 1 Real)

| Hora | Duração | Tempo Acumulado |
|------|---------|-----------------|
| 12 AM → 1 AM | 90s | 90s |
| 1 AM → 2 AM | 89s | 179s |
| 2 AM → 3 AM | 89s | 268s |
| 3 AM → 4 AM | 89s | 357s |
| 4 AM → 5 AM | 89s | 446s |
| 5 AM → 6 AM | 89s | **535s total** |

### 3.3 Cálculo de Hora Atual

```python
if tempo_jogo < 90:
    hora = 0  # 12 AM
else:
    hora = 1 + int((tempo_jogo - 90) // 89)  # 1-6 AM
```

---

## 4. VALIDAÇÃO DE AÇÕES POR CONTEXTO

### 4.1 Regras Implementadas

**Função `_executar_acao()` agora retorna `bool`:**
- `True` = ação válida (teve efeito)
- `False` = ação inválida (sem efeito)

### 4.2 Tabela de Validação

| Ação | Contexto Necessário | Válida? |
|------|---------------------|---------|
| `porta_esquerda` | Câmera fechada | ✅ / ❌ |
| `porta_direita` | Câmera fechada | ✅ / ❌ |
| `luz_esquerda` | Câmera fechada | ✅ / ❌ |
| `luz_direita` | Câmera fechada | ✅ / ❌ |
| `abrir_fechar_camera` | Qualquer | ✅ Sempre |
| `camera_1a` ... `camera_7` | Câmera aberta | ✅ / ❌ |
| `nada` | Qualquer | ✅ Sempre |

### 4.3 Exemplo de Código

```python
# Portas/luzes só funcionam fora da câmera
if nome_acao in ["porta_esquerda", "porta_direita", "luz_esquerda", "luz_direita"]:
    if self.camera_aberta:
        return False  # Ação inválida
    # ... executa ação
    
# Trocar câmera só funciona com câmera aberta
elif nome_acao.startswith("camera_"):
    if not self.camera_aberta:
        return False  # Ação inválida
    self.camera_ativa = acao - 5
```

---

## 5. COMPORTAMENTO REALISTA DAS LUZES

### 5.1 Auto-Desligamento

**ANTES:**
```python
if nome_acao == "luz_esquerda":
    self.luz_esq = not self.luz_esq  # Toggle permanente
```

**DEPOIS:**
```python
if nome_acao == "luz_esquerda":
    self.luz_esq = True
    self.luz_esq_timer = 1  # Desliga após 1 step

def _atualizar_luzes(self):
    if self.luz_esq_timer > 0:
        self.luz_esq_timer -= 1
        if self.luz_esq_timer == 0:
            self.luz_esq = False
```

**Benefício:** Comportamento igual ao jogo real (luzes são momentâneas).

### 5.2 Desligamento ao Abrir Câmera

```python
if nome_acao == "abrir_fechar_camera":
    self.camera_aberta = not self.camera_aberta
    if self.camera_aberta:
        # Desliga luzes (não fazem sentido no contexto)
        self.luz_esq = False
        self.luz_dir = False
        self.luz_esq_timer = 0
        self.luz_dir_timer = 0
```

---

## 6. SISTEMA DE RECOMPENSAS (VERSÃO FINAL)

### 6.1 Tabela Completa de Recompensas

| Evento | Recompensa | Justificativa |
|--------|------------|---------------|
| **Vitória (6 AM)** | **+1000.0** | Objetivo principal |
| **Morte** | **-500.0** | Penalidade máxima |
| **Sem energia** | **-500.0** | Equivalente a morte |
| **Sobreviver step (base)** | **+0.5** | Reduzido de +1.0 para evitar passividade |
| **Progresso temporal** | **+0.0 a +0.5** | Cresce linearmente (tempo/535 × 0.5) |
| **Ação inválida** | **0.0** | Neutra (sem efeito no jogo) |
| **Usar porta** | **-0.3** | Gasta energia |
| **Usar luz** | **-0.2** | Gasta energia |
| **Ambas portas fechadas** | **-1.0** | Desperdício crítico |
| **Usar câmera** | **+0.4** | Aumentado de +0.2 para incentivar |
| **Energia < 40%** | **-0.8** | Aumentado de -0.4 para urgência |
| **Energia < 20%** | **-1.5** | Aumentado de -0.8 para urgência crítica |
| **Limite mínimo** | **-2.0** | Evita instabilidade no treino |

### 6.2 Penalidade por Repetição (Lógica Especial)

**Portas e Luzes** (permite ligar/desligar rápido):
```python
if nome_acao in ["porta_esquerda", "porta_direita", "luz_esquerda", "luz_direita"]:
    if nome_acao == self.penultima_acao:  # Só penaliza na 3ª repetição
        recompensa -= 1.5
```

**Outras ações** (câmeras):
```python
else:
    if nome_acao == self.ultima_acao:  # Penaliza já na 2ª repetição
        recompensa -= 1.0
```

### 6.3 Exemplos de Cálculo

#### Exemplo 1: Ação válida simples (início do jogo)
```
Ação: Luz esquerda
Tempo: 0s (progresso = 0.0)

Cálculo:
  +0.5 (base)
  +0.0 (progresso)
  -0.2 (usar luz)
  = +0.3
```

#### Exemplo 2: Usar câmera (meio do jogo)
```
Ação: Câmera 1A
Tempo: 267s (progresso = 0.5)

Cálculo:
  +0.5 (base)
  +0.25 (progresso: 267/535 × 0.5)
  +0.4 (bônus câmera)
  = +1.15
```

#### Exemplo 3: Energia crítica + ambas portas
```
Ação: Porta esquerda
Tempo: 400s (progresso = 0.75)
Energia: 15%
Estado: Ambas portas fechadas

Cálculo:
  +0.5 (base)
  +0.375 (progresso: 400/535 × 0.5)
  -0.3 (usar porta)
  -1.0 (ambas portas)
  -1.5 (energia < 20%)
  = -1.925
  Limitado a -2.0
```

#### Exemplo 4: Spam de câmera
```
Step 1: Câmera 1A
  +0.5 + 0.0 + 0.4 = +0.9

Step 2: Câmera 1A (repetiu)
  +0.5 + 0.0 + 0.4 - 1.0 = 0.0

Step 3: Câmera 1A (repetiu 3x)
  +0.5 + 0.0 + 0.4 - 1.0 = 0.0
```

#### Exemplo 5: Ligar/desligar porta (válido)
```
Step 1: Porta esquerda (abre)
  +0.5 + 0.0 - 0.3 = +0.2

Step 2: Porta esquerda (fecha)
  +0.5 + 0.0 - 0.3 = +0.2  ✅ Sem penalidade!

Step 3: Luz esquerda
  +0.5 + 0.0 - 0.2 = +0.3
```

### 6.4 Comparação: Fazer Nada vs Agir

**Estratégia passiva (2140 steps = 535s):**
```
2140 × (+0.5 base + 0.25 progresso médio) = +1605 pontos
Resultado: Morte (não chegou ao 6 AM)
Recompensa final: +1605 - 500 = +1105
```

**Estratégia ativa (sobreviveu até 6 AM):**
```
2140 × (+0.5 base + 0.25 progresso + 0.2 câmera - 0.15 uso médio) = +1712 pontos
Resultado: Vitória
Recompensa final: +1712 + 1000 = +2712 ✅
```

**Conclusão:** Vitória vale ~2.5x mais que passividade.

---

## 7. POLICY MULTIMODAL

### 7.1 Arquitetura

**Arquivo:** `src/agent/multimodal_policy.py`

```python
class MultimodalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # CNN para imagem (84x84x1)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # MLP para estados (7 dimensões)
        self.fc_estados = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
        )
        
        # Combina features
        self.fc_combined = nn.Sequential(
            nn.Linear(cnn_output_dim + 32, 256),
            nn.ReLU(),
        )
```

### 7.2 Fluxo de Dados

```
Observação Dict
├─ "imagem" (84x84x1) ──→ CNN ──→ Features visuais (N dimensões)
└─ "estados" (7,)      ──→ MLP ──→ Features estruturadas (32 dimensões)
                                    ↓
                              Concatenação
                                    ↓
                              FC Combined (256 dimensões)
                                    ↓
                              Actor-Critic Networks
```

### 7.3 Uso no Treino

**Arquivo:** `src/agent/train.py`

```python
from src.agent.multimodal_policy import MultimodalExtractor

policy_kwargs = dict(
    features_extractor_class=MultimodalExtractor,
)

modelo = PPO(
    policy="MultiInputPolicy",  # Mudou de "CnnPolicy"
    env=env,
    policy_kwargs=policy_kwargs,
    # ... outros parâmetros
)
```

---

## 8. RESUMO DAS MUDANÇAS NO CÓDIGO

### 8.1 Arquivos Modificados

1. **`src/environment/fnaf_env.py`** (principal)
   - Observation space: Box → Dict
   - Novos estados: 7 dimensões
   - Sistema de energia realista
   - Sistema de tempo
   - Validação de ações
   - Luzes auto-desligam
   - Recompensas balanceadas
   - Comportamento energia = 0

2. **`src/agent/multimodal_policy.py`** (novo)
   - Extrator de features multimodal
   - CNN + MLP combinados

3. **`src/agent/train.py`**
   - Usa MultiInputPolicy
   - Importa MultimodalExtractor

4. **`main.py`**
   - Modo teste atualizado para exibir 7 estados

### 8.2 Novos Arquivos de Documentação

1. **`docs/ENERGIA_E_TEMPO.md`**
   - Explicação do sistema de energia
   - Fórmulas do FNAF 1 real
   - Sistema de tempo

2. **`docs/SISTEMA_RECOMPENSAS.md`** (este arquivo)
   - Documentação completa das recompensas
   - Exemplos de cálculo
   - Justificativas

---

## 9. EXPECTATIVAS DE APRENDIZADO

### 9.1 Fases do Treino

| Episódios | Comportamento Esperado | Recompensa Média |
|-----------|------------------------|------------------|
| 0-50 | Exploração aleatória, aprende ações válidas | -400 a -200 |
| 50-200 | Para de fazer spam, começa a usar câmera | -200 a 0 |
| 200-500 | Sobrevive 1-3 minutos consistentemente | 0 a +300 |
| 500-1000 | Desenvolve estratégias, sobrevive 3-5 min | +300 a +600 |
| 1000-2000 | Estratégia refinada, ocasionalmente 6 AM | +600 a +1000 |
| 2000+ | Taxa de vitória aumenta gradualmente | +1000+ |

### 9.2 Métricas de Sucesso

**Após 100 episódios, a IA deve:**
- ✅ Não repetir mesma ação >3x seguidas
- ✅ Usar câmera pelo menos 20% do tempo
- ✅ Sobreviver >60 segundos em média
- ✅ Não deixar ambas portas fechadas constantemente

**Após 500 episódios, a IA deve:**
- ✅ Sobreviver >180 segundos em média
- ✅ Economizar energia quando <40%
- ✅ Variar entre câmera e portas/luzes
- ✅ Taxa de vitória >1%

**Após 2000 episódios, a IA deve:**
- ✅ Sobreviver >300 segundos em média
- ✅ Taxa de vitória >10%
- ✅ Estratégia consistente e reproduzível

---

## 10. LIMITAÇÕES E TRABALHOS FUTUROS

### 10.1 Limitações Atuais

1. **Sem detecção de animatronics**
   - IA aprende padrões temporais, não reage a ameaças visuais
   - Pode fechar portas em momentos errados

2. **Sem dreno passivo por noite**
   - Night 2+: deveria ter dreno extra a cada X segundos
   - Simplificação para facilitar aprendizado inicial

3. **Sem penalidade do Foxy**
   - Foxy deveria drenar energia ao bater na porta
   - Não implementado (requer detecção visual)

4. **Espaço de ações grande (17 ações)**
   - Exploração lenta
   - Poderia usar curriculum learning

### 10.2 Melhorias Futuras

1. **Detecção de animatronics via template matching**
   - Adicionar ao vetor de estados: `[bonnie_porta, chica_porta, foxy_correndo]`
   - Recompensa por fechar porta quando animatronic está perto

2. **Curriculum learning**
   - Fase 1: Apenas 5 ações básicas
   - Fase 2: Adicionar câmeras gradualmente
   - Fase 3: Todas as 17 ações

3. **Imitation learning**
   - Coletar dados de gameplay humano
   - Pré-treinar com behavioral cloning
   - Refinar com PPO

4. **Dreno passivo e Foxy**
   - Implementar mecânicas completas do jogo
   - Aumentar fidelidade ao FNAF 1 real

---

## 11. CONCLUSÃO

As alterações implementadas transformaram o ambiente de **parcialmente observável** (apenas imagem) para **híbrido observável** (imagem + estados internos), com:

✅ **Mecânicas realistas** baseadas no FNAF 1 real
✅ **Validação de contexto** para ações
✅ **Recompensas balanceadas** que incentivam comportamento ativo
✅ **Estabilidade de treino** (limite mínimo, recompensa base reduzida)
✅ **Arquitetura multimodal** (CNN + MLP)

**O ambiente agora está pronto para treino efetivo**, com expectativa de convergência em 1000-2000 episódios para comportamento consistente e primeiras vitórias.

---

**Última atualização:** Sessão de melhorias completa
**Versão do ambiente:** 2.0 (Multimodal + Estados Internos)
