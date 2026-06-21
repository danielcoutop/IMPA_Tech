# Engine de Xadrez com Redes Neurais (CNN vs ResNet)

Aplicação web local para jogar xadrez contra uma engine gulosa (1-ply) que
usa suas redes treinadas no notebook `projeto.ipynb` para avaliar posições.

## Estrutura

```
chess_app/
├── backend/
│   ├── app.py            # Servidor Flask + lógica da engine
│   ├── models.py         # Arquiteturas ChessEvalCNN / ChessEvalResNet
│   ├── requirements.txt
│   └── weights/          # <- COLOQUE SEUS ARQUIVOS .pt AQUI
│       ├── best_model_baseline_cnn.pt
│       └── best_model_resnet.pt
└── frontend/
    └── index.html        # Interface (tabuleiro do seu xadrez.svg + JS)
```

## Passo 1 — Coloque os pesos treinados

Copie os dois arquivos `.pt` gerados no notebook (via `torch.save(model.state_dict(), ...)`)
para a pasta `backend/weights/`, com exatamente estes nomes:

- `best_model_baseline_cnn.pt`
- `best_model_resnet.pt`

(Esses são os mesmos nomes de arquivo usados no seu notebook nas células de
treino — se você salvou com outro caminho/nome, ajuste `WEIGHTS_PATHS` no
topo de `backend/app.py`.)

## Passo 2 — Instale as dependências

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Passo 3 — Rode o servidor

```bash
python app.py
```

Abra **http://127.0.0.1:5000** no navegador.

## Como funciona

### Input da rede (codificação do tabuleiro)
Reproduzi exatamente a função `FEN_para_tensor` do seu notebook:
um tensor `(15, 8, 8)` onde:
- Canais 0–5: peças brancas (peão, cavalo, bispo, torre, dama, rei)
- Canais 6–11: peças pretas (mesma ordem)
- Canal 12: 1.0 em todo o canal se for a vez das Brancas
- Canal 13: 1.0 se Brancas tiverem direito a roque (qualquer lado)
- Canal 14: 1.0 se Pretas tiverem direito a roque (qualquer lado)

### Output da rede e convenção de sinal
A coluna `Evaluation_clean` do seu dataset é a avaliação em centipawns na
**perspectiva das Brancas** (positivo = bom para Brancas), independentemente
de quem joga. Por isso, no `choose_engine_move` (em `app.py`), a engine
(que joga de Pretas) escolhe o lance que **minimiza** a saída da rede.

Se ao testar você notar que a engine joga de forma claramente invertida
(entrega peças, ignora xeque-mates a favor dela, etc.), troque a constante
no topo de `app.py`:

```python
BLACK_WANTS_MINIMUM = False  # em vez de True
```

### A engine (busca gulosa, 1-ply)
Para cada lance legal do `python-chess`, simula a posição resultante,
avalia com a rede neural selecionada e escolhe o melhor lance — sem
minimax, sem alpha-beta, exatamente como pedido.

### Troca de modelo em tempo real
O botão "CNN Simples" / "ResNet" no painel lateral chama
`POST /api/set_model`, que troca qual `.pt` é usado nos próximos lances da
engine (os modelos ficam em cache depois do primeiro carregamento).

### Frontend
O HTML usa exatamente os `<symbol>` do seu `xadrez.svg` (peões, torre, rei,
dama, bispo, cavalo, com as variáveis de cor `--cor_1` a `--cor_6`),
posicionados dinamicamente via JavaScript a partir do FEN retornado pelo
backend. Cliques nas peças mostram os lances legais em destaque; promoção
de peão abre um seletor de peça.

## Limitações conhecidas / próximos passos sugeridos
- Não há suporte a "arrastar e soltar" (apenas clique-clique), pois você
  pediu a versão mais simples possível — é fácil estender depois.
- A busca é greedy 1-ply, como solicitado; não há detecção de xeque-mate
  forçado em profundidade maior.
- O estado do jogo é guardado em memória no processo Flask (uso local,
  um usuário por vez).
