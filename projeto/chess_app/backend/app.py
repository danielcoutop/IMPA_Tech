"""
Backend Flask para a engine de xadrez com avaliação por rede neural.

Endpoints:
    GET  /                      -> serve o frontend (index.html)
    POST /api/new_game          -> reinicia o jogo, retorna estado inicial
                                    (aceita {"model", "depth", "time_limit"})
    GET  /api/state             -> retorna o FEN atual e status do jogo
    POST /api/player_move       -> recebe o lance do usuário (Brancas),
                                    responde com o lance da engine (Pretas)
    POST /api/set_model         -> troca o modelo ativo ("cnn" ou "resnet")
    GET  /api/legal_moves       -> lista os lances legais a partir do quadrado dado
    GET  /api/config            -> limites/padrões de profundidade e tempo p/ a UI

Como rodar:
    pip install -r requirements.txt
    python app.py
    -> abra http://127.0.0.1:5000

Pesos esperados (ajuste os caminhos em WEIGHTS_PATHS abaixo):
    weights/best_model_baseline_cnn.pt
    weights/best_model_resnet.pt

Sobre a busca:
    A rede neural funciona como função de avaliação estática (heurística de
    folha), exatamente como uma função de avaliação material/posicional
    clássica funcionaria em uma engine tradicional. Em volta dela, a engine
    faz uma busca minimax com poda alpha-beta e aprofundamento iterativo
    (iterative deepening): começa em profundidade 1 e vai aumentando até
    bater no limite de profundidade escolhido pelo usuário OU esgotar o
    tempo máximo configurado — o que vier primeiro. Isso garante que tanto
    o slider de "profundidade" quanto o de "tempo máximo" tenham efeito real
    sobre a força/velocidade da IA, e que sempre seja devolvido o melhor
    lance encontrado até aquele ponto (mesmo que o tempo acabe no meio de
    uma profundidade maior).
"""

import os
import threading
import time

import chess
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from models import load_model

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

WEIGHTS_PATHS = {
    "cnn": os.path.join(BASE_DIR, "weights", "best_model_baseline_cnn.pt"),
    "resnet": os.path.join(BASE_DIR, "weights", "best_model_resnet.pt"),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Limites/padrões expostos para a tela inicial (GET /api/config)
MIN_DEPTH = 1
MAX_DEPTH = 6
DEFAULT_DEPTH = 3

MIN_TIME_LIMIT = 0.5
MAX_TIME_LIMIT = 30.0
DEFAULT_TIME_LIMIT = 5.0

# Convenção do dataset: Evaluation_clean é a avaliação em centipawns na
# perspectiva das BRANCAS (positivo = bom p/ brancas). O minimax abaixo já
# assume isso nativamente: Brancas MAXIMIZAM o valor, Pretas MINIMIZAM.
# Se notar que a engine joga "ao contrário" (entrega peças, ignora capturas
# óbvias), inverta este valor para -1 — ele afeta tanto a raiz quanto toda
# a árvore de busca.
EVAL_SIGN = 1.0

MATE_SCORE = 100_000.0

app = Flask(__name__, static_folder=None)

# ----------------------------------------------------------------------------
# Estado do jogo (single-user / local). Protegido por lock por simplicidade.
# ----------------------------------------------------------------------------

_lock = threading.Lock()
_state = {
    "board": chess.Board(),
    "model_name": "cnn",
    "depth": DEFAULT_DEPTH,
    "time_limit": DEFAULT_TIME_LIMIT,
}

_models_cache = {}


def get_model(name: str):
    """Carrega (com cache) o modelo pedido."""
    if name not in _models_cache:
        path = WEIGHTS_PATHS.get(name)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"Pesos do modelo '{name}' não encontrados em '{path}'. "
                f"Copie o arquivo .pt salvo no notebook para essa pasta."
            )
        _models_cache[name] = load_model(name, path, device=DEVICE)
    return _models_cache[name]


# ----------------------------------------------------------------------------
# Codificação do tabuleiro (idêntica ao notebook: Dataset_tensor.FEN_para_tensor)
# ----------------------------------------------------------------------------

PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

# Valores usados só para ORDENAÇÃO de lances (MVV-LVA), não para avaliação.
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Reproduz FEN_para_tensor do notebook, mas recebendo um chess.Board."""
    tensor = np.zeros((15, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
        tensor[channel, row, col] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1.0

    return torch.from_numpy(tensor)


@torch.no_grad()
def evaluate_position(model, board: chess.Board) -> float:
    """Avaliação estática (centipawns, perspectiva das brancas)."""
    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    pred = model(x)
    return EVAL_SIGN * float(pred.item())


# ----------------------------------------------------------------------------
# Ordenação de lances (acelera MUITO a poda alpha-beta)
# ----------------------------------------------------------------------------

def order_moves(board: chess.Board, moves):
    """
    Ordena lances do mais promissor para o menos promissor, usando
    heurísticas baratas (sem chamar a rede): capturas (MVV-LVA), promoções
    e xeques primeiro. Isso aumenta drasticamente a eficácia da poda
    alpha-beta, permitindo profundidades maiores no mesmo tempo.
    """

    def score(move: chess.Move) -> int:
        s = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            # en passant: a peça capturada não está em to_square
            if victim is None and board.is_en_passant(move):
                victim_value = PIECE_VALUE[chess.PAWN]
            else:
                victim_value = PIECE_VALUE[victim.piece_type] if victim else 0
            attacker = board.piece_at(move.from_square)
            attacker_value = PIECE_VALUE[attacker.piece_type] if attacker else 0
            # MVV-LVA: prioriza capturar peça valiosa com peça barata
            s += 10_000 + victim_value * 10 - attacker_value
        if move.promotion:
            s += 9_000 + PIECE_VALUE.get(move.promotion, 0)
        if board.gives_check(move):
            s += 50
        return s

    return sorted(moves, key=score, reverse=True)


# ----------------------------------------------------------------------------
# Engine: minimax com poda alpha-beta + aprofundamento iterativo
# ----------------------------------------------------------------------------

class _TimeUp(Exception):
    pass


class SearchEngine:
    """
    Busca o melhor lance numa posição usando minimax + alpha-beta, com a
    rede neural como função de avaliação estática nas folhas.

    - Brancas MAXIMIZAM a avaliação (centipawns, perspectiva das brancas).
    - Pretas MINIMIZAM a mesma avaliação.
    - Aprofundamento iterativo: tenta profundidade 1, 2, 3... até `max_depth`,
      parando antes se `time_limit` (segundos) for atingido. Sempre devolve
      o melhor lance da última profundidade COMPLETADA.
    """

    def __init__(self, model, max_depth: int, time_limit: float):
        self.model = model
        self.max_depth = max(MIN_DEPTH, int(max_depth))
        self.time_limit = max(MIN_TIME_LIMIT, float(time_limit))
        self.nodes = 0
        self.deadline = None

    def _check_time(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise _TimeUp()

    @torch.no_grad()
    def _leaf_eval(self, board: chess.Board) -> float:
        self.nodes += 1
        return evaluate_position(self.model, board)

    def _terminal_eval(self, board: chess.Board, ply: int):
        """Avaliação para fim de jogo, com bônus por mate mais rápido."""
        if board.is_checkmate():
            # Quem está em xeque-mate é quem teria a vez (board.turn).
            # Se brancas estão em mate -> péssimo p/ brancas -> valor bem negativo.
            mate_value = MATE_SCORE - ply
            return -mate_value if board.turn == chess.WHITE else mate_value
        # Empates (afogamento, material insuficiente, repetição, 50 lances)
        return 0.0

    def _alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        self._check_time()

        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            return self._terminal_eval(board, ply)

        if depth == 0:
            return self._leaf_eval(board)

        maximizing = board.turn == chess.WHITE
        moves = order_moves(board, list(board.legal_moves))

        best = -float("inf") if maximizing else float("inf")

        for move in moves:
            board.push(move)
            try:
                value = self._alphabeta(board, depth - 1, alpha, beta, ply + 1)
            finally:
                board.pop()

            if maximizing:
                if value > best:
                    best = value
                alpha = max(alpha, value)
            else:
                if value < best:
                    best = value
                beta = min(beta, value)

            if alpha >= beta:
                break  # poda alpha-beta

        return best

    def search(self, board: chess.Board):
        """
        Executa o aprofundamento iterativo e devolve
        (melhor_lance, avaliacao, profundidade_alcancada).
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, None, 0

        maximizing = board.turn == chess.WHITE
        self.nodes = 0
        self.deadline = time.monotonic() + self.time_limit

        best_move = legal_moves[0]
        best_eval = None
        depth_reached = 0

        for depth in range(1, self.max_depth + 1):
            try:
                moves = order_moves(board, legal_moves)
                alpha, beta = -float("inf"), float("inf")
                current_best_move = None
                current_best_eval = -float("inf") if maximizing else float("inf")

                for move in moves:
                    board.push(move)
                    try:
                        value = self._alphabeta(board, depth - 1, alpha, beta, 1)
                    finally:
                        board.pop()

                    if maximizing:
                        if current_best_move is None or value > current_best_eval:
                            current_best_eval, current_best_move = value, move
                        alpha = max(alpha, value)
                    else:
                        if current_best_move is None or value < current_best_eval:
                            current_best_eval, current_best_move = value, move
                        beta = min(beta, value)

                # Profundidade completa: aceita o resultado e tenta ir mais fundo.
                best_move, best_eval = current_best_move, current_best_eval
                depth_reached = depth

                # Coloca o melhor lance encontrado primeiro na próxima iteração
                # (move ordering por resultado da iteração anterior).
                if best_move in legal_moves:
                    legal_moves.remove(best_move)
                    legal_moves.insert(0, best_move)

                # Mate encontrado: não há razão para aprofundar mais.
                if best_eval is not None and abs(best_eval) >= MATE_SCORE - 1000:
                    break

            except _TimeUp:
                # Tempo esgotado no meio desta profundidade: mantém o
                # melhor lance da última profundidade COMPLETA.
                break

        if best_eval is None:
            best_eval = self._leaf_eval(board)

        return best_move, best_eval, depth_reached


@torch.no_grad()
def choose_engine_move(board: chess.Board, model_name: str, depth: int, time_limit: float):
    """
    Escolhe o lance da engine usando minimax + alpha-beta + aprofundamento
    iterativo, respeitando o limite de profundidade e o tempo máximo
    configurados pelo usuário.

    Retorna (melhor_lance, avaliacao, stats) ou (None, None, stats) se não
    houver lances legais (xeque-mate / afogamento).
    """
    model = get_model(model_name)

    start = time.monotonic()
    engine = SearchEngine(model, max_depth=depth, time_limit=time_limit)
    best_move, best_eval, depth_reached = engine.search(board)
    elapsed = time.monotonic() - start

    stats = {
        "depth_reached": depth_reached,
        "nodes": engine.nodes,
        "time_s": round(elapsed, 2),
    }

    return best_move, best_eval, stats


# ----------------------------------------------------------------------------
# Helpers de serialização
# ----------------------------------------------------------------------------

def board_status(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "draw_material"
    if board.can_claim_threefold_repetition():
        return "draw_repetition"
    if board.is_check():
        return "check"
    return "ongoing"


def serialize_state(board: chess.Board, last_move: chess.Move = None, engine_eval: float = None,
                     search_stats: dict = None):
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "status": board_status(board),
        "game_over": board.is_game_over(),
        "last_move": last_move.uci() if last_move else None,
        "engine_eval": engine_eval,
        "search_stats": search_stats,
    }


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


# ----------------------------------------------------------------------------
# Rotas da API
# ----------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/config", methods=["GET"])
def config():
    return jsonify({
        "min_depth": MIN_DEPTH,
        "max_depth": MAX_DEPTH,
        "default_depth": DEFAULT_DEPTH,
        "min_time_limit": MIN_TIME_LIMIT,
        "max_time_limit": MAX_TIME_LIMIT,
        "default_time_limit": DEFAULT_TIME_LIMIT,
    })


@app.route("/api/new_game", methods=["POST"])
def new_game():
    data = request.get_json(silent=True) or {}

    model_name = data.get("model", _state.get("model_name", "cnn"))
    if model_name not in WEIGHTS_PATHS:
        return jsonify({"error": "Modelo inválido. Use 'cnn' ou 'resnet'."}), 400

    try:
        depth = int(data.get("depth", DEFAULT_DEPTH))
    except (TypeError, ValueError):
        depth = DEFAULT_DEPTH
    depth = clamp(depth, MIN_DEPTH, MAX_DEPTH)

    try:
        time_limit = float(data.get("time_limit", DEFAULT_TIME_LIMIT))
    except (TypeError, ValueError):
        time_limit = DEFAULT_TIME_LIMIT
    time_limit = clamp(time_limit, MIN_TIME_LIMIT, MAX_TIME_LIMIT)

    try:
        get_model(model_name)  # erro claro e imediato se faltar o .pt
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    with _lock:
        _state["board"] = chess.Board()
        _state["model_name"] = model_name
        _state["depth"] = depth
        _state["time_limit"] = time_limit
        return jsonify(serialize_state(_state["board"]))


@app.route("/api/state", methods=["GET"])
def state():
    with _lock:
        return jsonify(serialize_state(_state["board"]))


@app.route("/api/set_model", methods=["POST"])
def set_model():
    data = request.get_json(force=True) or {}
    name = data.get("model")
    if name not in WEIGHTS_PATHS:
        return jsonify({"error": "Modelo inválido. Use 'cnn' ou 'resnet'."}), 400

    try:
        get_model(name)  # força o carregamento agora, retorna erro claro se faltar o .pt
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    with _lock:
        _state["model_name"] = name

    return jsonify({"ok": True, "model": name})


@app.route("/api/legal_moves", methods=["GET"])
def legal_moves():
    square_name = request.args.get("square")
    if not square_name:
        return jsonify({"error": "Parâmetro 'square' é obrigatório."}), 400

    with _lock:
        board = _state["board"]
        try:
            square = chess.parse_square(square_name)
        except ValueError:
            return jsonify({"error": "Quadrado inválido."}), 400

        moves = [
            m.uci() for m in board.legal_moves
            if m.from_square == square
        ]
    return jsonify({"moves": moves})


@app.route("/api/player_move", methods=["POST"])
def player_move():
    """
    Recebe o lance das Brancas (usuário), aplica, e se o jogo não acabou,
    calcula e aplica a resposta das Pretas (engine) via busca alpha-beta
    com aprofundamento iterativo, respeitando profundidade e tempo máximo
    escolhidos na tela inicial.

    Body esperado: {"move": "e2e4"} (formato UCI), ou
                   {"from": "e2", "to": "e4", "promotion": "q"}
    """
    data = request.get_json(force=True) or {}

    with _lock:
        board = _state["board"]
        model_name = _state["model_name"]
        depth = _state.get("depth", DEFAULT_DEPTH)
        time_limit = _state.get("time_limit", DEFAULT_TIME_LIMIT)

        if board.is_game_over():
            return jsonify({"error": "O jogo já terminou."}), 400
        if board.turn != chess.WHITE:
            return jsonify({"error": "Não é a vez das Brancas."}), 400

        # Monta o lance UCI a partir do payload
        uci_move = data.get("move")
        if not uci_move:
            from_sq = data.get("from")
            to_sq = data.get("to")
            promotion = data.get("promotion", "")
            if not from_sq or not to_sq:
                return jsonify({"error": "Informe 'move' (UCI) ou 'from'/'to'."}), 400
            uci_move = f"{from_sq}{to_sq}{promotion}"

        try:
            player_chess_move = chess.Move.from_uci(uci_move)
        except ValueError:
            return jsonify({"error": "Lance em formato UCI inválido."}), 400

        if player_chess_move not in board.legal_moves:
            return jsonify({"error": "Lance ilegal."}), 400

        board.push(player_chess_move)

        if board.is_game_over():
            return jsonify({
                "player_move": player_chess_move.uci(),
                "engine_move": None,
                "state": serialize_state(board, player_chess_move),
            })

        # Vez da engine (Pretas)
        try:
            engine_move, engine_eval, search_stats = choose_engine_move(
                board, model_name, depth, time_limit
            )
        except FileNotFoundError as e:
            board.pop()  # desfaz o lance do jogador para manter consistência
            return jsonify({"error": str(e)}), 404

        if engine_move is not None:
            board.push(engine_move)

        return jsonify({
            "player_move": player_chess_move.uci(),
            "engine_move": engine_move.uci() if engine_move else None,
            "state": serialize_state(
                board, engine_move or player_chess_move, engine_eval, search_stats
            ),
        })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)