# no-more-jumpscares

Projeto de agente por reforço (PPO) que interage com o jogo
Five Nights at Freddy's (FNAF1) via captura de tela e controle
de mouse/teclado. O objetivo é treinar uma IA para sobreviver
às noites do jogo automaticamente.

**Visão Geral**

- **Entrada:** screenshots do jogo (captura de janela)
- **Ações:** cliques nas posições mapeadas (portas, luzes, câmeras)
- **Algoritmo:** PPO via `stable_baselines3`

**Pré-requisitos**

- **Sistema operacional:** Windows (testado)
- **Jogo:** Five Nights at Freddy's instalado (obrigatório)
- **Python:** 3.10+ (recomenda-se 3.10–3.13)
- **Dependências:** instale via `pip install -r requirements.txt`
- **Recomendado:** escala de exibição do Windows em 100% para
  garantir consistência nas coordenadas de clique

Instalação rápida

```bash
# criar e ativar virtualenv (Windows)
python -m venv venv
venv\Scripts\activate

# instalar dependências
pip install -r requirements.txt
```

Configuração importante — modo janela e coordenadas

- O jogo DEVE estar em modo janela (não em fullscreen). No FNAF
  isso costuma ser alternado com **ALT+ENTER**. Antes de executar
  qualquer script de calibração ou treino, pressione **ALT+ENTER**
  para colocar o jogo em modo janela.
- As coordenadas de clique são lidas a partir da posição da janela
  em modo janela. Ou seja: sempre cadastre as coordenadas com o
  jogo em modo janela (ALT+ENTER) e com a janela posicionada
  como pretende usá‑la durante o treino.

Como cadastrar coordenadas e imagens de referência

1. Abra o jogo e pressione **ALT+ENTER** (modo janela).
2. Posicione a janela do jogo na tela e confirme escala 100%.
3. Rode o script de calibragem para obter coordenadas do mouse:

```bash
# mostra coordenadas em tempo real (mova o mouse sobre cada botão)
python -m src.utils.calibrar

# captura imagem de morte (abra a tela de Game Over e execute)
python -m src.utils.calibrar morte

# captura imagem de vitória (quando aparecer o 6 AM)
python -m src.utils.calibrar vitoria
```

4. Copie os pares `x, y` impressos e atualize o dicionário `COORDS`
   em [src/environment/fnaf_env.py](src/environment/fnaf_env.py) para
   que as ações cliquem nas posições corretas do seu jogo.

Referências visuais

- As imagens de referência são carregadas de `src/utils/referencias/`.
  O código espera encontrar as imagens de morte/6AM nesta pasta —
  verifique os nomes esperados em `src/environment/fnaf_env.py` e, se
  necessário, renomeie as imagens geradas por `calibrar`.

Comandos principais

- Testar reset/observação (verifica se a captura funciona):

```bash
python main.py teste
```

- Rodar treino (PPO):

```bash
python main.py treino
```

Exportar logs para MongoDB

- Instale o driver do MongoDB para Python:

```bash
pip install pymongo
```

- Configure no `.env` (copie de `.env.example`):
  - `PC` (identificador da maquina)
  - `MONGO_URI`
  - `MONGO_DATABASE`
  - `MONGO_COLLECTION`
- Execute o treino para gerar logs:

```bash
python main.py treino
```

    O treino gera dados em:
    - `logs/treino.log` ou `logs/treino.log.txt` (log textual por episodio)
    - `logs/PPO_1/events.out.tfevents...` (TensorBoard)

Formato padrao do log textual (mesmo estilo de `logs/treino.log.txt`):

```text
============================================================
Treino iniciado
============================================================
Ep    1 | MORTE    | Passos:    455 | Recompensa:    114.8 | Taxa vitória: 0.0%
pc0 | Ep    2 | MORTE    | Passos:    510 | Recompensa:     37.5 | Taxa vitória: 0.0%
```

Observacoes:

- O prefixo `pc0 |` e opcional em cada linha de episodio.
- Cada bloco `Treino iniciado` e tratado como uma sessao de treino no parser.
- O script gera no JSON os atributos de raiz `pc` e `sessao_treino_id`.

- Envie para o MongoDB:

```bash
python scripts/enviar_logs_mongodb.py
```

O script envia um JSON com:

- `pc`: identificador do PC
- `sessao_treino_id`: identificador unico para agrupar a mesma execucao
- `dados_log`: registros do log (CSV, log textual ou TensorBoard)

Exemplos uteis:

```bash
# Validar payload sem enviar para o banco
python scripts/enviar_logs_mongodb.py --dry-run --print-json

# Forcar uma fonte de log especifica
python scripts/enviar_logs_mongodb.py --source logs_analysis/episodes.csv

# Usar explicitamente o log no formato .txt
python scripts/enviar_logs_mongodb.py --source logs/treino.log.txt
```

Observações e dicas

- Durante o treino, segure **F12** para pausar a execução (o
  callback checa essa tecla e pausa o loop de treino enquanto
  estiver pressionada).
- Use `tensorboard --logdir logs` para visualizar métricas do
  `stable_baselines3` (recompensa média por episódio, loss, etc.).
- Se muitos episódios terminam com morte imediata (passos = 1),
  verifique as coordenadas em `COORDS` e confirme o modo janela.
- Se a captura não encontrar a janela pelo título, ajuste o nome
  procurado em `GameCapture.focar_janela()` em
  [src/utils/capture.py](src/utils/capture.py).

Arquivos úteis

- Código do ambiente: [src/environment/fnaf_env.py](src/environment/fnaf_env.py)
- Scripts de utilitários: [src/utils/calibrar.py](src/utils/calibrar.py)
- Captura de tela: [src/utils/capture.py](src/utils/capture.py)
- Logs de treino: `logs/` (TensorBoard em `logs/`)

Suporte
Se quiser, eu posso: gerar um checklist automatizado de calibração,
abrir o TensorBoard para você ou adicionar um script que salva as
coordenadas automaticamente em um arquivo de configuração.

--
README gerado automaticamente pelo assistente.
