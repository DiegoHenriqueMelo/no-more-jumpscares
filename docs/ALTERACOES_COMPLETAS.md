# Alterações do ambiente — histórico técnico

Documento de referência para as mudanças feitas no ambiente de RL do FNAF 1.
Cobre duas sessões de trabalho: expansão do espaço de observação e refinamento
de recompensas/correção de bugs.

---

## Espaço de observação

O espaço original era um único `Box(84, 84, 1)` — só a imagem. O agente não tinha
acesso a nenhum estado interno do jogo: não sabia se a porta estava fechada, quanto
de energia tinha, se a câmera estava aberta. Ele precisaria aprender tudo isso só
pelos pixels, o que é possível mas lento e frágil.

A mudança foi trocar para `Dict` com dois campos:

- `imagem`: o frame capturado em escala de cinza, redimensionado para 84×84
- `estados`: vetor float32 de 7 dimensões com os estados internos normalizados

Os 7 estados são, nessa ordem: `porta_esq`, `porta_dir`, `luz_esq`, `luz_dir`,
`camera_aberta`, `camera_ativa / 11.0`, `energia / 100.0`.

Isso exigiu mudar a política para `MultiInputPolicy` e criar um extrator de features
customizado (`MultimodalExtractor` em `src/agent/multimodal_policy.py`). A rede
processa a imagem por uma CNN de 3 camadas conv e os estados por um MLP simples
(`Linear(7→32)`), concatena os dois e passa por uma camada final de 256 unidades.

---

## Sistema de energia

Antes não havia simulação de energia no lado Python — o agente só via o medidor
pelo pixel. Agora o ambiente rastreia `self.energia` em tempo real usando
`time.perf_counter()` para calcular o `dt` entre atualizações.

O consumo por segundo segue a lógica do jogo:

```
usage = 1  # base (ventilador sempre ligado)
usage += porta_esq + porta_dir + luz_esq + luz_dir + camera_aberta
usage = min(usage, 4)
consumo = usage * 0.1  # %/s
```

Com todos os sistemas ativos (4 barras), a energia acaba em ~250s — bem antes dos
535s da noite completa. Isso cria pressão real para o agente economizar.

Quando `energia <= 0`, o ambiente desliga tudo e aguarda a morte (Freddy demora
alguns segundos para aparecer quando a energia acaba), retornando recompensa −500.

---

## Validação de ações por contexto

`_executar_acao` passou a retornar `bool` indicando se a ação teve efeito.

Portas e luzes só funcionam com câmera **fechada** — no jogo real o painel de
controle fica oculto quando a câmera está aberta. Trocar de câmera só funciona
com câmera **aberta**. Ações inválidas retornam `False` e recebem recompensa 0
(nem positiva nem negativa, só neutras).

---

## Luzes com duração de 1 passo

Antes as luzes eram toggle permanente. No jogo real elas são momentâneas — você
segura o botão e solta. A implementação nova liga a luz e seta um timer de 1 step;
no próximo `_atualizar_luzes()` ela apaga sozinha. Ao abrir a câmera, ambas as
luzes são forçadas para off (o painel não existe nesse contexto).

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

## Correções de timing

### Timer de energia

`self.ultimo_update_energia` era definido no início do `reset()`, antes dos
~35 segundos de sleep necessários para o jogo reiniciar. Isso fazia com que o
primeiro `_atualizar_energia()` do episódio calculasse um `dt` de ~35s,
consumindo energia indevidamente logo no passo 1.

Correção: o timer agora é definido no final do `reset()`, junto com
`episode_start_time`, imediatamente antes de capturar a observação inicial.

### Guard de detecção de morte

O guard que ignora detecção nos primeiros N passos (para o jogo terminar de
transicionar da tela de Game Over) estava em 10 passos (~2.5s). Insuficiente —
o FNAF pode demorar mais para limpar a tela. Aumentado para 30 passos (~7.5s).
Mesmo ajuste aplicado em `_detectar_vitoria`.

### Timer do episódio no log

O SB3 `DummyVecEnv` chama `env.reset()` de forma **síncrona** dentro de
`env.step()` quando `done=True`, antes de devolver o controle ao PPO. O callback
`_on_step` só dispara depois disso — ou seja, ~35s após o fim real do episódio.
O campo "Tempo" no log estava inflado em ~35s por episódio. Episódios de 1 passo
apareciam como "0.00 min" porque início e fim do timer caíam no mesmo `_on_step`.

Correção: o ambiente mede `tempo_real = time.perf_counter() - episode_start_time`
no momento em que o episódio termina e passa pelo `info`. O callback usa esse
valor diretamente, sem depender de wall-clock próprio.

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

Os 5 segundos correspondem à mecânica real do Foxy: qualquer câmera aberta nesse
intervalo paralisa o avanço dele. Ficar com câmera fechada além disso é
objetivamente perigoso.

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

A penalidade é proporcional ao déficit: `deficit * 0.02` por passo. Se o agente
está 20% abaixo do esperado para o horário, paga −0.4 por passo. Isso distribui
a pressão de economia de energia ao longo de toda a noite, não só no final.

### Tabela final de recompensas

| Evento | Valor |
|--------|-------|
| Morte / energia zerada | −500 |
| Sobreviver (6 AM) | +1000 |
| Ação inválida | 0 |
| Progresso na noite | 0 a +0.5 (linear) |
| Ação repetida | −1.0 (ou −1.5 na 3ª vez para portas/luzes) |
| Ambas as portas fechadas | −1.0/passo |
| Uso de luz | −0.2 |
| Câmera inativa >20 passos | −0.05 × excesso (máx −1.0) |
| Energia abaixo do esperado | −0.02 × déficit |
| Limite mínimo por passo | −2.0 |

---

## Parâmetros de treino

`gamma` foi aumentado de 0.99 para **0.995**. O motivo é numérico: uma noite tem
~970 passos (535s ÷ 0.55s/passo). Com gamma=0.99, o +1000 de sobreviver vale
`0.99^970 ≈ 0.06` no passo inicial — virtualmente zero. O agente não conseguia
"enxergar" que sobreviver valia a pena. Com 0.995, vale ~7.7, 130× mais impactante.

---

## Arquivos modificados

- `src/environment/fnaf_env.py` — todas as mudanças do ambiente
- `src/agent/multimodal_policy.py` — novo, extrator CNN + MLP
- `src/agent/train.py` — gamma, MultiInputPolicy, timer do log corrigido
- `src/utils/capture.py` — método `arrastar_para` com duração
- `.env` / `.env.example` — novas variáveis de drag da câmera
