"""
Backend Flask para a engine de xadrez com avaliação por rede neural.

Endpoints:
    GET  /                      -> serve o frontend (index.html)
    POST /api/new_game          -> reinicia o jogo, retorna estado inicial
    GET  /api/state             -> retorna o FEN atual e status do jogo
    POST /api/player_move       -> recebe o lance do usuário (Brancas),
                                    responde com o lance da engine (Pretas)
    POST /api/set_model         -> troca o modelo ativo ("cnn" ou "resnet")
    GET  /api/legal_moves       -> lista os lances legais a partir do quadrado dado

Como rodar:
    pip install -r requirements.txt
    python app.py
    -> abra http://127.0.0.1:5000

Pesos esperados (ajuste os caminhos em WEIGHTS_PATHS abaixo):
    weights/best_model_baseline_cnn.pt
    weights/best_model_resnet.pt
"""

import os
import threading

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

# Convenção do dataset: Evaluation_clean é a avaliação em centipawns na
# perspectiva das BRANCAS (positivo = bom p/ brancas), independente de quem
# joga. A engine joga de Pretas, então ela quer MINIMIZAR essa saída.
# Se notar que a engine joga "ao contrário" (entrega peças, ignora capturas
# óbvias), inverta este valor para -1.
BLACK_WANTS_MINIMUM = True

app = Flask(__name__, static_folder=None)

# ----------------------------------------------------------------------------
# Estado do jogo (single-user / local). Protegido por lock por simplicidade.
# ----------------------------------------------------------------------------

_lock = threading.Lock()
_state = {
    "board": chess.Board(),
    "model_name": "cnn",
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
    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    pred = model(x)
    return float(pred.item())


# ----------------------------------------------------------------------------
# Engine: busca gulosa (greedy) de 1-ply
# ----------------------------------------------------------------------------

@torch.no_grad()
def choose_engine_move(board: chess.Board, model_name: str):
    """
    A engine joga de Pretas. Para cada lance legal, simula a posição
    resultante, avalia com a rede e escolhe o melhor lance (mínimo, pois
    o output é centipawns na perspectiva das brancas).
    Retorna (melhor_lance, avaliacao, lista_de_avaliacoes) ou (None, None, [])
    se não houver lances legais (xeque-mate / afogamento).
    """
    model = get_model(model_name)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, None, []

    best_move = None
    best_eval = None
    evaluations = []

    for move in legal_moves:
        board.push(move)
        score = evaluate_position(model, board)
        board.pop()

        evaluations.append({"move": move.uci(), "eval": score})

        if best_eval is None:
            best_move, best_eval = move, score
        else:
            is_better = (score < best_eval) if BLACK_WANTS_MINIMUM else (score > best_eval)
            if is_better:
                best_move, best_eval = move, score

    return best_move, best_eval, evaluations


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


def serialize_state(board: chess.Board, last_move: chess.Move = None, engine_eval: float = None):
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "status": board_status(board),
        "game_over": board.is_game_over(),
        "last_move": last_move.uci() if last_move else None,
        "engine_eval": engine_eval,
    }


# ----------------------------------------------------------------------------
# Rotas da API
# ----------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/new_game", methods=["POST"])
def new_game():
    with _lock:
        _state["board"] = chess.Board()
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
    calcula e aplica a resposta das Pretas (engine).
    Body esperado: {"move": "e2e4"} (formato UCI), ou
                   {"from": "e2", "to": "e4", "promotion": "q"}
    """
    data = request.get_json(force=True) or {}

    with _lock:
        board = _state["board"]
        model_name = _state["model_name"]

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
            engine_move, engine_eval, _ = choose_engine_move(board, model_name)
        except FileNotFoundError as e:
            board.pop()  # desfaz o lance do jogador para manter consistência
            return jsonify({"error": str(e)}), 404

        if engine_move is not None:
            board.push(engine_move)

        return jsonify({
            "player_move": player_chess_move.uci(),
            "engine_move": engine_move.uci() if engine_move else None,
            "state": serialize_state(board, engine_move or player_chess_move, engine_eval),
        })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)