# Visão computacional para observabilidade

Este documento registra a discussão sobre adicionar **detecção visual dos
animatrônicos** ao projeto — a ideia inicialmente chamada de "reconhecimento
facial" — como forma de melhorar a observabilidade do agente. É um registro de
projeto: captura a ideia, os contrapontos, as decisões em aberto e as
recomendações. **Nada aqui foi implementado ainda.**

> Status: discussão de design. Nenhuma das opções abaixo foi decidida nem
> aplicada ao código. As "possibilidades" são exatamente isso — possibilidades.

---

## A ideia original

A proposta: a IA abre a câmera, e ao identificar o animatrônico ali, um
algoritmo define um **peso de perigo** — quão ameaçado o agente está. Com isso
ele aprenderia sobre a dificuldade do jogo.

### Ajuste de nome

O que foi descrito não é *reconhecimento facial* (biometria de rosto humano). É
**detecção de objeto / reconhecimento dos animatrônicos** (Bonnie, Chica, Foxy,
Freddy) nas câmeras e nas portas. Isso importa porque o projeto **já usa essa
família de técnica**: o [src/environment/fnaf_env.py](../src/environment/fnaf_env.py)
usa `cv2.matchTemplate` para detectar morte, vitória e se a câmera está aberta.
A ideia é o mesmo motor, apontado para um alvo novo: "tem um animatrônico
nesta sala/porta?".

Também não é preciso "abrir uma câmera" nova — o agente já captura a janela do
jogo a cada step (`_capturar_janela`). A detecção rodaria sobre esse frame que
já existe.

---

## Por que a ideia é boa: observabilidade é o gargalo

O ponto mais fraco do projeto hoje está documentado em
[ALEM_DO_RL.md](ALEM_DO_RL.md) e [MELHORIAS_PROJETO_ATUAL.md](MELHORIAS_PROJETO_ATUAL.md):
**observabilidade parcial**.

- O agente enxerga só um frame **84×84 em escala de cinza**. Numa câmera escura
  do FNAF, um animatrônico nesse tamanho é quase invisível — a CNN tem que
  adivinhar.
- Dar ao agente um sinal explícito ("tem vulto na porta esquerda", "Foxy está no
  estágio 3") transforma um problema difícil (POMDP) em algo mais perto de um
  MDP — facilita a *atribuição de crédito* (ligar "porta aberta + vulto na
  entrada → morte").

### O "Marco Polo"

O FNAF é, por design, um jogo de informação escondida: os animatrônicos se movem
**quando você não está olhando**. Hoje o agente joga meio às cegas, e isso
desacelera o aprendizado.

Ponto importante: **visão computacional não elimina o Marco Polo — ela melhora o
"ouvido".** A detecção não dá o mapa inteiro; o agente continua vendo uma câmera
por vez (como um humano). O que muda é a *qualidade de cada olhada*:

- **Hoje:** olha a Cam 2A, recebe 84×84 cinza, e *adivinha* se aquele borrão é o
  Bonnie. Olhada cara e pouco confiável.
- **Com detecção:** olha a Cam 2A e tem certeza. Cada olhada vira informação
  sólida.

O que *eliminaria* o Marco Polo de vez seria **ler a memória do `fnaf.exe`**
(abordagem `pymem` do [ALEM_DO_RL.md](ALEM_DO_RL.md)) — mas isso quebra a
premissa "a IA aprende pela percepção visual, como um humano". A visão é o
meio-termo: acelera bastante sem trair a premissa.

### Por que a lentidão dói tanto aqui

O treino roda em **tempo real**: cada noite dura ~9 min de relógio. Não há
milhões de steps baratos como num simulador — cada episódio é **caro**. Logo, a
métrica que importa é *sample efficiency*: quanto o agente aprende por episódio.
É por isso que observabilidade pesa mais aqui do que em RL comum: cada amostra
cara precisa ser o mais informativa possível.

---

## Como usar o peso de perigo

Há dois caminhos, e a diferença é tudo:

1. **Como observação (recomendado):** o resultado da detecção entra no vetor
   `estados`. O agente continua aprendendo a estratégia sozinho — você só dá
   olhos melhores.
2. **Como recompensa (cuidado):** penalizar "porta aberta com animatrônico" (o
   [ALEM_DO_RL.md](ALEM_DO_RL.md) cita esse exemplo). Funciona, mas **embute a
   estratégia humana** na recompensa e abre risco de reward hacking.

**Regra:** observação = aprendizado preservado; reward/regra = você programou a
estratégia.

---

## Robustez: a imagem do FNAF é propositalmente enganosa

A memória temporal já está sendo atacada em **outro ramo do projeto**, onde o
**RecurrentPPO** (PPO com camada LSTM, do `sb3-contrib`) foi aplicado para dar
persistência de memória ao agente — e os primeiros resultados **parecem
promissores**. Isso muda (e simplifica) o desenho da visão: como a persistência
temporal vem do LSTM, a camada de visão **não precisa ser perfeita frame a
frame**. O trabalho dela vira:

> "Dado *este* frame, qual minha melhor estimativa de perigo — e quão confiável
> ela é?"

Por isso a saída da detecção deve ter **dois canais**:

- `peso_perigo` ∈ [0,1] — quão perigoso parece.
- `confianca` ∈ [0,1] — quão confiável é a leitura (estática/glitch/preto →
  confiança baixa).

Sem o canal de confiança, um frame de estática vira um falso "porta vazia". Com
ele, a estática diz "não sei agora" e o LSTM segura a última leitura boa.

### O reframe que resolve a "cara alterada"

**Não fazer reconhecimento facial. Fazer detecção de ocupação (anomalia).**

Para decidir fechar a porta, o agente não precisa reconhecer o rosto (que muda
com pose/distorção). Ele só precisa saber: *o corredor iluminado está vazio ou
tem algo ali?* Isso se detecta por **desvio da referência do vazio**:

- Referência = corredor iluminado **vazio** (imagem consistente).
- Detecção = o quanto o frame atual difere dessa referência.
- Vulto grande = diferença alta = perigo — **não importa qual animatrônico nem
  se a cara está distorcida**, porque você detecta "tem algo que não deveria
  estar", não um rosto específico.

Isso é naturalmente robusto às distorções. `matchTemplate` de uma cara
específica é a opção mais frágil exatamente onde o FNAF engana.

### Tratando estática / glitch / tela preta

1. **Threshold + voto de múltiplos frames:** captura 2–3 frames rápidos e tira a
   mediana. Estática é aleatória por frame; um animatrônico real é consistente.
2. **Detector de estática → canal de confiança:** estática tem assinatura
   característica (variância altíssima, sem estrutura). Quando detecta, baixa a
   `confianca` em vez de cuspir falso negativo.
3. **A estática também é sinal de jogo:** glitch pesado costuma acompanhar
   movimento de animatrônico. Dá pra alimentar o nível de estática como feature —
   mas isso é refinamento, não MVP.

### Quando você quer identificar *qual* (ex: estágio do Foxy)

Para a porta, ocupação basta. Para distinguir casos (estágio do Foxy na Cam 1C,
qual animatrônico está onde), há dois caminhos:

| Abordagem | Robustez a estática/distorção | Custo |
|---|---|---|
| `matchTemplate` (banco de templates por pose) | Baixa — quebra nas distorções | Zero treino, encaixa no repo |
| CNN classificadora pequena com *data augmentation* (ruído, brilho, distorção) | Alta — aprende as invariâncias | Precisa de screenshots rotulados + treino supervisionado |

O augmentation é o que mata o problema da "cara alterada": treina mostrando de
propósito as versões distorcidas e ruidosas. Como é supervisionado (rótulos
baratos), é muito mais barato que esperar o PPO descobrir sozinho em tempo real.

---

## A pergunta filosófica: isso vira um script de if/else?

Depende **de onde** você pluga o peso de perigo:

- **Como observação:** *não* vira if/else. A detecção é um **sensor**, não a
  decisão. Mesma diferença entre dar um LIDAR a um carro autônomo e programar o
  volante na mão. O agente ainda precisa **aprender** que "algo na porta → fechar
  ajuda". A decisão segue aprendida.
- **Como reward ou regra de ação:** *aí sim* vira if/else disfarçado. Evitar.

**Custo honesto:** mesmo como observação, isso *desloca onde o aprendizado
acontece*. A percepção passa a ser engenharia/treino supervisionado, e o RL só
aprende a *agir*. A tese deixa de ser "a IA aprende a **perceber e agir** a
partir de pixels" e vira "a IA aprende a **agir** dada uma percepção assistida".
É uma afirmação mais estreita — vale assumir conscientemente.

**Contraponto:** um humano não aprende a reconhecer o Bonnie via reward — o
córtex visual já entrega "tem um vulto ali". Então um módulo de percepção é, em
certo sentido, *mais* parecido com humano do que obrigar uma CNN minúscula a
aprender visão a partir de sinais de morte esparsos. Dá pra defender os dois
lados.

---

## Imparcial: isso resolve o problema atual?

O problema é "o aprendizado não anda". As causas estruturais são três, e a
detecção ataca **uma**:

| Gargalo | A detecção resolve? |
|---|---|
| Observabilidade parcial (cego no escuro) | **Sim**, diretamente |
| Custo de amostra (tempo real, poucos episódios) | **Não** — sensor não dá mais episódios |
| Design de reward / bugs residuais | **Não** |

Se o gargalo dominante for **número de amostras** (comum em jogo de tempo real),
o agente pode ter visão perfeita e *ainda* não convergir. Logo: a detecção
**remove um handicap conhecido**, mas **não há garantia** de que conserta o
aprendizado. A forma honesta de saber é o **experimento A/B** (com vs sem
detecção), que vira resultado de TCC.

---

## Possibilidade considerada: reduzir o vetor / cortar a CNN

> Estas são **possibilidades discutidas**, não decisões. Em particular, **não
> houve decisão de cortar a CNN** — apenas exploramos o que aconteceria.

### Reduzir o vetor de estados e jogar tudo na CNN

Avaliado e **não recomendado**. Energia, tempo, estado de portas/câmera são
sinais baratos, confiáveis e de baixa dimensão. Forçar a CNN a inferir energia a
partir de 84×84 pixels devora amostras — e amostra é o recurso mais escasso.
Encolher o vetor deixa o aprendizado **mais difícil**. A direção certa é manter o
vetor e *acrescentar* a detecção.

### Cortar a CNN do loop de RL (possibilidade)

A ideia explorada: tirar a imagem/CNN da política e decidir só com
(vetor de estados) + (detecções). Importante separar **dois significados de
"CNN"**:

| | Treinada pelo reward (RL)? | Nesta possibilidade |
|---|---|---|
| **CNN da política** (dentro do `MultimodalExtractor`) | Sim | Seria cortada |
| **Visão de percepção** (reconhece padrões pré-definidos) | Não — é sensor | Mantida |

Ou seja: cortar a CNN **do loop de reward**, mas a parte de reconhecer padrão
continua existindo como **percepção/sensor**, fora do RL.

**Prós (imparcial):**

- **Muito mais sample-efficient.** Um MLP pequeno sobre ~10–15 features
  estruturadas aprende ordens de magnitude mais rápido que uma CNN sobre pixels.
  Ataca o gargalo de amostra de frente.
- **A CNN provavelmente agregava pouco** — frame escuro 84×84 com reward esparso
  é quase ruído. Se a detecção já extrai o que importa, a CNN vira parâmetro
  difícil de treinar.
- **Casa perfeitamente com o RecurrentPPO** — LSTM sobre features estruturadas
  resolve o "lembrar o perigo da câmera X enquanto o tablet está fechado".

**Contras (imparcial):**

- **O teto passa a ser o conjunto de features.** Sem pixels na rede, não há
  fallback: o que a detecção não extrai, o agente não percebe.
- **É o maior golpe na tese "aprende a ver"** — elimina a percepção visual do
  agente. Continua sendo RL (não é if/else), mas a visão vira 100% engenheirada.
- **A robustez da detecção vira crítica** — sem CNN para compensar.

### A "CNN de percepção" pode ser de dois tipos

1. **CV clássica (`matchTemplate` / ocupação / anomalia):** tecnicamente **não é
   CNN** — é correlação do OpenCV. Reconhece padrão, mas **não aprende**. Zero
   treino, porém frágil onde o FNAF engana.
2. **CNN classificadora supervisionada (congelada):** *é* uma CNN reconhecendo
   padrões pré-definidos. **Aprende**, mas por treino **supervisionado, offline,
   com rótulos** — não pelo reward do RL. Depois fica congelada como sensor.

Implicação para a tese:

- CV clássica → percepção é engenharia pura, 0 aprendizado.
- CNN supervisionada → **dois sistemas que aprendem** (visão supervisionada + RL),
  nenhum é if/else. Melhor narrativa: a IA *aprende a enxergar* (supervisionado,
  robusto via augmentation) **e** *aprende a jogar* (RL), sem pagar custo de
  amostra de treinar visão em tempo real.

**Caveat que não some:** "padrões pré-definidos" é o teto. O agente só percebe o
que você definiu/treinou.

### O caveat mais importante para o TCC

> Quanto mais completa a percepção engenheirada, mais trivial fica a política
> aprendida — e mais o *resultado* se parece com o if/else que se queria evitar,
> mesmo tendo sido aprendido.

No FNAF1, dado o estado quase completo, a política ótima é quase uma regra
simples. O agente não a escreveu (convergiu pra ela), mas o "olha, aprendeu!"
perde impacto quando o estado torna a resposta óbvia. Tensão central:

- **CNN + observabilidade parcial:** problema difícil, resultado impressionante —
  mas pode não convergir no orçamento de amostras.
- **Vetor + detecção, sem CNN:** problema fácil, converge provável — mas o feito
  é menos impressionante.

Não há escolha neutra: é um trade entre *converge* e *impressiona*.

---

## A câmera balança sozinha

Ponto técnico que separa o MVP do resto:

- **Portas (visão do escritório com a luz): não balançam.** São estáticas. O MVP
  de ocupação **não é afetado** por isso.
- **Câmeras: balançam involuntariamente** entre dois pontos, com pausa breve em
  cada (e demora pra chegar). Esse é o desafio da fase 2.

Como dar robustez sem depender da pausa:

1. **Detecção tolerante a translação.** O balanço é quase só deslocamento.
   `matchTemplate` já desliza o template por uma *região* e acha o melhor encaixe
   onde quer que esteja — então uma panorâmica leve é absorvida. (Diff
   pixel-a-pixel contra um frame fixo quebra com o balanço; evitar.)
2. **Não esperar a pausa.** Amostrar enquanto está na câmera, mesmo em
   movimento. O RecurrentPPO funde vários frames, então cada relance imperfeito
   contribui. É assim que se "garante o dado mesmo se movendo": parar de caçar o
   frame perfeito.
3. **O balanço é a favor.** Mostra a sala de dois ângulos — mais cobertura, não
   menos. Calibrar/treinar com referências dos dois pontos + do meio.
4. **CNN treinada:** data augmentation com translação e blur cobre o balanço
   automaticamente — mais um motivo para preferir modelo treinado a template
   rígido *nas câmeras*.

---

## Régua das abordagens

```
CNN pura (pixels)            → aprende ver + decidir   (puro, faminto por amostra)
CNN + detecção               → aprende decidir, visão assistida, com fallback
Vetor + detecção (sem CNN)   → aprende decidir, visão 100% engenheirada, sem fallback
if/else nas detecções        → não aprende nada   ← NÃO é o que se propõe
```

---

## Recomendações e próximos passos

1. **MVP nas portas = detecção de ocupação por desvio do vazio**, saída
   `(peso_perigo, confianca)` por lado, **como observação** (não reward). Robusto
   à distorção, encaixa no padrão de visão do
   [fnaf_env.py](../src/environment/fnaf_env.py), roda no frame full-res que o
   `_capturar_observacao` já captura. Zero treino. Precisa de uma referência do
   corredor **vazio iluminado** de cada lado (fluxo do
   `python -m src.utils.calibrar`). Sem a referência, deixar o recurso inerte
   (fallback neutro, como o template de câmera faz hoje) — nada quebra.
2. **Foxy / identificação por câmera = fase 2**, com CNN classificadora +
   augmentation, se a ocupação simples não bastar (e por causa do balanço).
3. **Manter o vetor de estados** — não encolher.
4. **Rodar o experimento A/B** (com vs sem detecção; e, se quiser explorar a
   possibilidade, vetor-only vs vetor+CNN). Mesma reward, mesmo RecurrentPPO, só
   muda a entrada. Mede sample efficiency e win rate. Os dois desfechos são
   publicáveis e viram resultado de TCC.

### Acoplamento a considerar (quando for implementar)

Mudar o vetor `estados` (hoje 8 dims) toca quatro lugares:

- [src/environment/fnaf_env.py](../src/environment/fnaf_env.py) — `observation_space`
  e `_capturar_observacao`.
- [src/agent/multimodal_policy.py](../src/agent/multimodal_policy.py) — o
  `nn.Linear(8, 32)` está hardcoded.
- [src/agent/behavioral_cloning.py](../src/agent/behavioral_cloning.py) — monta o
  vetor a partir do dataset (frames são 84×84, então as novas dims de detecção
  usariam valor neutro no BC).
- [scripts/smoke_test.py](../scripts/smoke_test.py) — asserta `estados.shape == (8,)`.

Lembrete: qualquer mudança no espaço de observação **invalida modelos antigos**
(muda a dimensão de entrada da rede). Como o treino já recomeça do zero
(pasta `modelos/` vazia), isso é compatível com a decisão atual do projeto.
