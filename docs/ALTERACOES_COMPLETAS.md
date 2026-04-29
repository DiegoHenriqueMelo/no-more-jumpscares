# Alterações do ambiente — histórico técnico

Documento de referência para as mudanças feitas no ambiente de RL do FNAF 1.

---

## Espaço de observação

O espaço original era um único `Box(84, 84, 1)` — só a imagem. O agente não tinha
acesso a nenhum estado interno do jogo: não sabia se a porta estava fechada, quanto
de energia tinha, se a câmera estava aberta. Ele precisaria aprender tudo isso só
pelos pixels, o que é possível mas lento e frágil.

A mudança foi trocar para `Dict` com dois campos:

- `imagem`: o frame capturado em escala de cinza, redimensionado para 84×84
- `estados`: vetor float32 de **8 dimensões** com os estados internos normalizados

Os 8 estados são, nessa ordem: `porta_esq`, `porta_dir`, `luz_esq`, `luz_dir`,
`camera_aberta`, `camera_ativa / 11.0`, `energia / 100.0`,
`tempo_real_ep / 535.0`.

A 8ª dimensão usa `time.perf_counter() - episode_start_time` (tempo real desde o
início do episódio), não tempo simulado. Isso é importante porque o tempo real por
step (~0.7s com animações) é maior que `STEP_DELAY` (~0.35s), então o tempo
simulado acumularia apenas ~40–50% do progresso real da noite. Com tempo real, o
agente sabe em qual hora do jogo está de fato, o que é informação relevante para
calibrar o comportamento (Freddy quase não se move antes das 3AM; a pressão dos
animatrônicos cresce progressivamente).

`episode_start_time` é definido **após** o último sleep do `reset()`, imediatamente
antes do primeiro step. Isso garante que o timer reflita somente o tempo de
gameplay, sem incluir o overhead de reset (~35s).

A rede processa a imagem por uma CNN de 3 camadas conv e os estados por um MLP
(`Linear(8→32)`), concatena os dois e passa por uma camada final de 256 unidades.

---

## Sistema de energia

O consumo de energia é separado em parcela passiva (fixo) e ativa (proporcional ao
número de itens ligados), com base nas taxas medidas do FNAF1 Night 1:

```
consumo_por_segundo = 0.104 + itens_ativos * 0.100
itens_ativos = min(porta_esq + porta_dir + luz_esq + luz_dir + camera_aberta, 3)
```

O dreno é aplicado por step usando `STEP_DELAY` como proxy de tempo:

```python
self.energia -= consumo_por_segundo * STEP_DELAY
```

Isso mantém a simulação determinística e desacoplada do wall-clock real, evitando
que variações de latência (animações, captura de tela) distorçam o modelo de
energia. A taxa de 0.104 + 0.100 por item foi calibrada contra o jogo real usando
o script `src/utils/simular_energia.py`.

Quando `energia <= 0`, o ambiente desliga tudo e aguarda a morte via template
matching — o Freddy demora alguns segundos para aparecer após a energia zerar, e
encerrar o episódio imediatamente corromperia o estado do reset.

---

## Validação de ações por contexto

`_executar_acao` retorna `bool` indicando se a ação teve efeito.

Portas e luzes só funcionam com câmera **fechada** — no jogo real o painel de
controle fica oculto quando a câmera está aberta. Trocar de câmera só funciona
com câmera **aberta**. Ações inválidas retornam `False` e recebem recompensa −0.5.

---

## Luzes — comportamento toggle persistente

As luzes funcionam como toggle: a primeira pressão liga, a segunda desliga.
Uma vez ativas, permanecem ligadas até:

1. O agente pressionar o mesmo botão novamente (desliga)
2. A câmera ser aberta (força off ambas)
3. A energia zerar (força off ambas)

Apenas uma luz pode estar ativa por vez: ligar a esquerda desliga a direita
automaticamente e vice-versa.

Fisicamente, o botão da luz é pressionado e solto a cada step imediatamente
antes da captura de observação. Isso garante que a imagem observe o corredor
iluminado enquanto a luz está ativa, sem manter o mouse clicado entre steps
(o que causaria conflitos com outras ações).

---

## Arrasto do mouse para a câmera

O botão de abrir/fechar a prancheta de câmeras no FNAF só aparece quando o mouse
**se move** em direção à parte inferior da tela — um teleporte direto para as
coordenadas não aciona a animação.

A solução: antes de clicar no botão, o mouse teleporta para `CAMERA_DRAG_PIXELS`
acima da coordenada do botão e então desliza suavemente até ele usando
`pyautogui.moveTo(x, y, duration=CAMERA_DRAG_DURATION, tween=easeInOutQuad)`.
Dois parâmetros configuráveis no `.env`:

```
FNAF_CAMERA_DRAG_PIXELS=80      # distância de partida acima do botão
FNAF_CAMERA_DRAG_DURATION=0.15  # duração do arrasto em segundos
```

Se o botão não aparecer, aumentar pixels para 120–150. Se o jogo não reconhecer
o movimento, aumentar a duração para 0.2–0.3.

---

## Sincronização de estado (câmera e portas)

O ambiente mantém estado interno (`camera_aberta`, `porta_esq`, `porta_dir`) que
pode desincronizar do jogo real por atrasos de animação, perda de clique ou
qualquer ruído externo. Três mecanismos passivos corrigem isso sem injetar cliques
fora da política do agente:

### Template matching — câmera

A cada 3 steps (fora do cooldown de abertura), `_camera_aberta_por_template()`
roda `cv2.matchTemplate` contra o indicador "YOU" no mapa de câmeras. Uma única
leitura discordante é descartada; duas leituras consecutivas discordantes corrigem
`self.camera_aberta`. Isso evita falsos positivos durante a animação de transição.

O template de referência (`src/utils/referencias/camera_aberta.png`) é gerado uma
vez pelo script de calibração:

```bash
python -m src.utils.calibrar camera_aberta
```

### Pre-click — portas

Imediatamente antes de toggler uma porta, o ambiente lê o pixel do botão e compara
com o estado interno. Se a cor dominante (verde = fechada, vermelho = aberta)
discordar, o estado interno é corrigido **antes** do toggle. Isso garante que a
ação aplicada (abrir ou fechar) corresponde ao que o agente pretendia.

### Pós-clique — portas

Após clicar em uma porta, `_verificar_botao_porta()` verifica se a cor dominante
do botão **inverteu** (verde→vermelho ou vermelho→verde). Se não tiver invertido
em até 3 tentativas, sincroniza o estado interno pela cor lida — sem reverter às
cegas. A verificação exige inversão de cor dominante (não apenas variação de delta),
o que elimina falsos positivos por ruído de iluminação.

### Sync passivo — luz → porta

Quando o agente pressiona uma luz, o ambiente aproveita a captura de tela já feita
para ler o pixel do botão de **porta do mesmo lado**. Se a cor dominante indicar
estado diferente do interno, corrige silenciosamente. Nenhum clique extra é
emitido.

### Log de desyncs por episódio

Ao fim de cada episódio, `_escrever_log_desyncs()` appenda uma linha em
`logs/desyncs.log`:

```
Ep    1 | steps   584 | desfecho: morte     | SYNC camera:   2 | SYNC porta:   0 | porta falha:   1
```

- **SYNC camera**: quantas vezes o template corrigiu `camera_aberta`
- **SYNC porta**: quantas vezes pre-click ou luz-sync corrigiram uma porta
- **porta falha**: quantas vezes `_verificar_botao_porta` não confirmou o toggle
  após 3 tentativas (indica clique perdido pelo jogo)

---

## Correções de timing

### Energia por STEP_DELAY em vez de wall-clock

A versão anterior calculava `dt = time.perf_counter() - ultimo_update_energia`,
o que causava dois problemas: (1) o primeiro step do episódio calculava um `dt`
de ~35s (todo o sleep do reset), consumindo energia indevidamente; (2) variações
de latência entre steps (animações de porta, drag da câmera) distorciam o modelo
de energia, fazendo-o drenar 2–3× mais rápido que o esperado.

Correção: `_atualizar_energia` usa `STEP_DELAY` como dt fixo. A energia drena
de forma determinística e proporcional ao número de steps, independente do
wall-clock. `ultimo_update_energia` foi removido.

### Timer do episódio

`episode_start_time` era definido antes do último sleep do `reset()` (~20s de
antecedência). Isso inflava `tempo_real` no log e deslocava os checkpoints de hora
em ~20s. Corrigido: o timer é definido **depois** do sleep, imediatamente antes
do primeiro step.

### Guard de detecção de morte e vitória

O guard que ignora detecção nos primeiros N passos (para o jogo terminar de
transicionar da tela de Game Over) estava em 10 passos (~2.5s). Insuficiente —
o FNAF pode demorar mais para limpar a tela. Aumentado para 120 passos (~30s).
Mesmo ajuste aplicado em `_detectar_vitoria`.

---

## Checkpoint de hora

Milestones de sobrevivência são recompensados ao atingir cada hora do jogo (1AM a
6AM). O checkpoint usa **tempo real do episódio** (`time.perf_counter() -
episode_start_time`) em vez de tempo simulado, pois o step real dura ~0.7s enquanto
`STEP_DELAY` é ~0.35s — tempo simulado refletiria apenas metade do progresso real.

O bônus é proporcional à energia disponível vs. a esperada para aquele horário:

```python
ratio = min(self.energia / e_cp, 1.5)
bonus = max(ratio * 50.0, 5.0)  # floor 5, máx 75
```

| Energia no checkpoint | Ratio | Bônus |
|-----------------------|-------|-------|
| 0% | 0.0 | +5 (floor) |
| Metade do esperado | 0.5 | +25 |
| Exatamente o esperado | 1.0 | +50 |
| 1.5× o esperado | 1.5 | +75 (cap) |

O cap em 1.5× garante que mesmo os 6 checkpoints no máximo (6 × 75 = 450) não
superem a penalidade de morte (−500), evitando que o agente aprenda a conservar
energia passivamente ignorando ameaças.

---

## Função de recompensa

### O que mudou e por quê

**Recompensa base de "nada"** era +0.5 por passo. Com um gamma baixo e recompensas
pequenas por passo, a estratégia mais segura era não fazer nada — afinal, qualquer
ação poderia ter penalidade. Zerado para 0.0.

**Spam de "nada"** não era penalizado (havia `and nome_acao != "nada"` no check
de repetição). Removida essa exceção — repetir "nada" agora custa −1.0 igual a
qualquer outra ação repetida.

**Penalidade por fechar porta** (−0.3 por ato) desincentivava fechar mesmo quando
necessário. Removida. O custo de energia de ter a porta fechada já é a penalidade
natural — o agente aprende a abrir quando o perigo passa.

**Bônus fixo por câmera** (+0.4 por qualquer ação de câmera) gerava spam de troca
de câmera sem contexto. Substituído por uma penalidade de **inatividade**:

```python
if self.passos_sem_camera > 20:          # ~5s sem abrir câmera
    excesso = self.passos_sem_camera - 20
    recompensa -= min(excesso * 0.05, 1.0)
```

**Survival bonus por step** garante que sobreviver mais é sempre melhor do que
morrer cedo, mesmo com penalidades acumulando. A recompensa base por passo é +0.5,
crescendo até +1.0 linearmente conforme o progresso da noite:

```python
recompensa = 0.5 + (self.tempo_jogo / 535.0) * 0.5
```

**Penalidades de energia por threshold fixo** (<20%, <40%) foram substituídas por
uma penalidade graduada baseada nos thresholds recomendados do jogo por horário:

| Horário | Energia esperada |
|---------|-----------------|
| 12 AM   | 100%            |
| 1 AM    | 85%             |
| 2 AM    | 60%             |
| 3 AM    | 40%             |
| 4 AM    | 25%             |
| 5 AM    | 15%             |
| 6 AM    | 5%              |

A penalidade é proporcional ao déficit: `deficit * 0.02` por passo.

### Tabela final de recompensas

| Evento | Valor |
|--------|-------|
| Morte | −500 |
| Sobreviver (6 AM) | +1000 |
| Ação inválida | −0.5 |
| Sobrevivência por step | +0.5 a +1.0 (cresce com progresso) |
| Checkpoint de hora (1AM–6AM) | +5 a +75 (proporcional à margem de energia) |
| Ação repetida | −1.0 (ou −1.5 na 3ª vez para portas/luzes) |
| Ambas as portas fechadas | −1.0/passo |
| Uso de luz | −0.2 |
| Câmera inativa >20 passos | −0.05 × excesso (máx −1.0) |
| Energia abaixo do esperado | −0.02 × déficit |
| Limite mínimo por passo | −2.0 |

---

## Parâmetros de treino

`gamma` foi aumentado de 0.99 para **0.995**. Com `STEP_DELAY=0.35` e step real
~0.7s, uma noite tem ~700 steps. Com gamma=0.99, o +1000 de sobreviver vale
`0.99^700 ≈ 0.0009` no passo inicial — virtualmente zero. Com 0.995, vale ~0.03,
dando ao agente um sinal real para "enxergar" que sobreviver a noite vale a pena.

---

## Previsão de aprendizado

Com `n_steps=2048` e episódios de ~600–800 steps, cada atualização de política
cobre aproximadamente 3 episódios. Os seguintes marcos são esperados com
treinamento contínuo:

| Timesteps | Atualizações | Comportamento esperado |
|-----------|-------------|------------------------|
| 0–50k | 0–25 | Política aleatória. Recompensa −800 a −600. |
| 50k–200k | 25–100 | Câmera começa a ser priorizada (penalidade de inatividade). Recompensa −600 a −400. |
| 200k–500k | 100–244 | Economia de energia emergindo. Checkpoint 3AM possível. Recompensa −400 a −100. |
| 500k–1M | 244–500 | Estratégia mais consistente. Fechamento de portas aprendido. Primeiras vitórias isoladas. |
| 1M–3M | 500–1500 | Vitórias periódicas. Taxa crescendo para 10–30%. |
| 3M+ | 1500+ | Taxa de vitória >50% possível se política convergida. |

Esses marcos assumem treinamento em um único PC. Com múltiplos PCs rodando em
paralelo e modelos combinados periodicamente, os timesteps efetivos escalam
linearmente com o número de máquinas.

As estimativas podem variar dependendo da qualidade da captura de imagem
(iluminação, resolução da janela do jogo) e da variância introduzida pelo
comportamento dos animatrônicos.

---

## Arquivos modificados

- `src/environment/fnaf_env.py` — todas as mudanças do ambiente
- `src/agent/multimodal_policy.py` — extrator CNN + MLP (Linear 7→8 para 8ª dimensão)
- `src/agent/train.py` — gamma, MultiInputPolicy, log de episódio
- `src/utils/capture.py` — método `arrastar_para` com duração
- `src/utils/calibrar.py` — comando `camera_aberta` para gerar o template de referência
- `src/utils/simular_energia.py` — utilitário para calibrar e verificar taxas de energia
- `.env` / `.env.example` — variáveis de drag da câmera e coordenadas das ações
