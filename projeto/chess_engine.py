"""
chess_engine.py
===============
Engine de xadrez com busca em árvore genérica + interface visual via pygame.

Busca: Minimax + Alpha-Beta com Iterative Deepening limitado por tempo (4 s).
       A eval_fn é completamente genérica — troca o modelo, tudo funciona.

Controles:
  Clique  — selecionar / mover peça
  R       — reiniciar
  U       — desfazer último par de lances
  ESC     — sair
"""

import sys
import time
import threading
import chess
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Tensorização — fiel ao Dataset_tensor.FEN_para_tensor do notebook
# ---------------------------------------------------------------------------

_PIECE_TO_CHANNEL = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def fen_para_tensor(fen: str) -> torch.Tensor:
    board = chess.Board(fen)
    t = np.zeros((15, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        t[_PIECE_TO_CHANNEL[(piece.piece_type, piece.color)], row, col] = 1.0
    if board.turn == chess.WHITE:
        t[12, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
        t[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
        t[14, :, :] = 1.0
    return torch.tensor(t).unsqueeze(0)  # (1, 15, 8, 8)


# ---------------------------------------------------------------------------
# 2. Modelo CNN Baseline — mesma arquitetura do notebook (célula 2.2)
# ---------------------------------------------------------------------------

class ChessEvalCNN(nn.Module):
    def __init__(self, in_channels=15, conv_channels=(32, 64, 64), hidden_dim=128, dropout=0.2):
        super().__init__()
        c1, c2, c3 = conv_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 8 * 8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.head(self.conv_block(x)).squeeze(-1)


def carregar_modelo(checkpoint_path, device):
    model = ChessEvalCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def make_eval_fn(model: nn.Module, device: torch.device):
    """
    Retorna eval_fn(fen) -> float genérica.
    Para trocar de modelo, basta passar outro nn.Module aqui.
    """
    @torch.no_grad()
    def eval_fn(fen: str) -> float:
        x = fen_para_tensor(fen).to(device)
        return float(model(x).item())
    return eval_fn


# ---------------------------------------------------------------------------
# 3. Busca — Minimax + Alpha-Beta + Iterative Deepening com limite de tempo
# ---------------------------------------------------------------------------

INF = float("inf")

class _Timeout(Exception):
    pass


def _minimax(board, depth, alpha, beta, maximizing, eval_fn, deadline):
    """Minimax recursivo com alpha-beta e verificação de deadline."""
    # Checar tempo a cada nó folha (barato)
    if depth == 0:
        if time.monotonic() > deadline:
            raise _Timeout()
        return eval_fn(board.fen())

    # Estados terminais — verificar antes de iterar lances
    if board.is_checkmate():
        # Quem está para jogar levou mate → ruim para quem maximiza
        return -INF if maximizing else INF
    if board.is_stalemate() or board.is_insufficient_material() or \
       board.can_claim_fifty_moves() or board.is_repetition(3):
        return 0.0

    if maximizing:
        value = -INF
        for move in board.legal_moves:
            board.push(move)
            value = max(value, _minimax(board, depth - 1, alpha, beta, False, eval_fn, deadline))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = INF
        for move in board.legal_moves:
            board.push(move)
            value = min(value, _minimax(board, depth - 1, alpha, beta, True, eval_fn, deadline))
            board.pop()
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


class Engine:
    """
    Engine genérica. Recebe qualquer eval_fn(fen) -> float.

    Usa Iterative Deepening: começa em depth_min e aprofunda até depth_max
    ou até esgotar time_limit segundos — o que vier primeiro.
    O resultado da profundidade completa mais recente é sempre devolvido,
    garantindo que nunca retorna None por timeout.
    """

    def __init__(self, eval_fn, depth_min: int = 2, depth_max: int = 8,
                 time_limit: float = 4.0, color: bool = chess.BLACK):
        self.eval_fn    = eval_fn
        self.depth_min  = depth_min
        self.depth_max  = depth_max
        self.time_limit = time_limit
        self.color      = color

    def escolher_lance(self, board: chess.Board):
        """
        Devolve o melhor lance encontrado dentro do orçamento de tempo.
        Garante retorno não-None enquanto houver lances legais.
        """
        legal = list(board.legal_moves)
        if not legal:
            return None

        # Caso trivial: único lance possível
        if len(legal) == 1:
            return legal[0]

        maximizing_root = (self.color == chess.WHITE)
        deadline        = time.monotonic() + self.time_limit

        best_move = legal[0]   # fallback seguro (nunca None)
        best_val  = -INF if maximizing_root else INF

        for depth in range(self.depth_min, self.depth_max + 1):
            candidate_move = best_move
            candidate_val  = -INF if maximizing_root else INF

            try:
                for move in legal:
                    board.push(move)
                    val = _minimax(
                        board,
                        depth=depth - 1,
                        alpha=-INF,
                        beta=INF,
                        maximizing=not maximizing_root,
                        eval_fn=self.eval_fn,
                        deadline=deadline,
                    )
                    board.pop()

                    if maximizing_root and val > candidate_val:
                        candidate_val, candidate_move = val, move
                    elif not maximizing_root and val < candidate_val:
                        candidate_val, candidate_move = val, move

                # Profundidade concluída sem timeout → promover resultado
                best_move = candidate_move
                best_val  = candidate_val

            except _Timeout:
                # Tempo esgotado no meio desta profundidade:
                # manter o best_move da profundidade anterior (já confiável)
                board_copy_clean(board)   # garante que não há push pendente
                break

        return best_move


def board_copy_clean(board: chess.Board):
    """
    Segurança: se um _Timeout interrompeu no meio de um push,
    o board pode estar com um lance não desempilhado.
    Isso NÃO acontece porque o try/except captura antes do pop(),
    mas mantemos esta função como no-op documentada por clareza.
    """
    pass   # python-chess mantém o stack íntegro; pop() ocorre antes do raise


# ---------------------------------------------------------------------------
# 4. Desenho de peças sem depender de Unicode/fonte do sistema
#    Cada peça é um conjunto de primitivas pygame — funciona em qualquer OS.
# ---------------------------------------------------------------------------

def _desenhar_peca_pygame(surface, pg, piece_type, is_white, x, y, size):
    """Desenha uma peça usando primitivas pygame. x,y = canto superior esquerdo da casa."""
    cx = x + size // 2
    cy = y + size // 2
    s  = size

    BRANCO  = (240, 235, 210)
    PRETO_P = (50,  45,  40)
    CONTORNO = PRETO_P if is_white else BRANCO
    CORPO    = BRANCO   if is_white else PRETO_P

    def circle(surf, cor, center, r, width=0):
        pg.draw.circle(surf, cor, center, r, width)

    def rect(surf, cor, rx, ry, rw, rh, border=0):
        pg.draw.rect(surf, cor, (rx, ry, rw, rh), border)

    def poly(surf, cor, points, width=0):
        pg.draw.polygon(surf, cor, points, width)

    def ellipse(surf, cor, ex, ey, ew, eh, width=0):
        pg.draw.ellipse(surf, cor, (ex, ey, ew, eh), width)

    u = s // 10   # unidade base

    if piece_type == chess.PAWN:
        # Cabeça
        circle(surface, CORPO,    (cx, cy - 2*u), 2*u)
        circle(surface, CONTORNO, (cx, cy - 2*u), 2*u, max(1, u//2))
        # Base
        rect(surface, CORPO,    cx - 3*u, cy + u,   6*u, 2*u)
        rect(surface, CONTORNO, cx - 3*u, cy + u,   6*u, 2*u, max(1, u//2))
        # Pescoço
        rect(surface, CORPO,    cx - u,   cy - u,   2*u, 3*u)
        rect(surface, CONTORNO, cx - u,   cy - u,   2*u, 3*u, max(1, u//2))

    elif piece_type == chess.KNIGHT:
        # Corpo estilizado (cabeça de cavalo)
        pts_corpo = [
            (cx - 2*u, cy + 3*u),
            (cx - 3*u, cy - u),
            (cx - u,   cy - 4*u),
            (cx + 2*u, cy - 4*u),
            (cx + 3*u, cy - 2*u),
            (cx + u,   cy),
            (cx + 3*u, cy + 3*u),
        ]
        poly(surface, CORPO,    pts_corpo)
        poly(surface, CONTORNO, pts_corpo, max(1, u//2))
        # Orelha
        circle(surface, CORPO,    (cx + 2*u, cy - 4*u), u)
        # Olho
        eye_cor = PRETO_P if is_white else BRANCO
        circle(surface, eye_cor,  (cx + u, cy - 2*u), max(1, u//2))
        # Base
        rect(surface, CORPO,    cx - 3*u, cy + 3*u, 6*u, 2*u)
        rect(surface, CONTORNO, cx - 3*u, cy + 3*u, 6*u, 2*u, max(1, u//2))

    elif piece_type == chess.BISHOP:
        # Corpo
        pts = [
            (cx,       cy - 4*u),
            (cx + 2*u, cy),
            (cx + 3*u, cy + 3*u),
            (cx - 3*u, cy + 3*u),
            (cx - 2*u, cy),
        ]
        poly(surface, CORPO,    pts)
        poly(surface, CONTORNO, pts, max(1, u//2))
        # Bola do topo
        circle(surface, CORPO,    (cx, cy - 4*u), u)
        circle(surface, CONTORNO, (cx, cy - 4*u), u, max(1, u//2))
        # Base
        rect(surface, CORPO,    cx - 3*u, cy + 3*u, 6*u, 2*u)
        rect(surface, CONTORNO, cx - 3*u, cy + 3*u, 6*u, 2*u, max(1, u//2))

    elif piece_type == chess.ROOK:
        # Torre
        rect(surface, CORPO,    cx - 3*u, cy - 3*u, 6*u, 6*u)
        rect(surface, CONTORNO, cx - 3*u, cy - 3*u, 6*u, 6*u, max(1, u//2))
        # Ameias
        for dx in (-2, 0, 2):
            rect(surface, CORPO,    cx + dx*u - u, cy - 4*u, 2*u, u + 2)
            rect(surface, CONTORNO, cx + dx*u - u, cy - 4*u, 2*u, u + 2, max(1, u//3))
        # Janela
        slot_cor = PRETO_P if is_white else (80, 70, 60)
        rect(surface, slot_cor, cx - u, cy - u, 2*u, 2*u)
        # Base
        rect(surface, CORPO,    cx - 3*u, cy + 3*u, 6*u, 2*u)
        rect(surface, CONTORNO, cx - 3*u, cy + 3*u, 6*u, 2*u, max(1, u//2))

    elif piece_type == chess.QUEEN:
        # Coroa
        pts_coroa = [
            (cx - 3*u, cy + u),
            (cx - 3*u, cy - 3*u),
            (cx - u,   cy - u),
            (cx,       cy - 4*u),
            (cx + u,   cy - u),
            (cx + 3*u, cy - 3*u),
            (cx + 3*u, cy + u),
        ]
        poly(surface, CORPO,    pts_coroa)
        poly(surface, CONTORNO, pts_coroa, max(1, u//2))
        # Pérolas no topo
        for px_dot, py_dot in [
            (cx - 3*u, cy - 3*u),
            (cx,       cy - 4*u),
            (cx + 3*u, cy - 3*u),
        ]:
            circle(surface, CORPO,    (px_dot, py_dot), u)
            circle(surface, CONTORNO, (px_dot, py_dot), u, max(1, u//2))
        # Base
        rect(surface, CORPO,    cx - 3*u, cy + u,   6*u, 2*u)
        rect(surface, CONTORNO, cx - 3*u, cy + u,   6*u, 2*u, max(1, u//2))
        rect(surface, CORPO,    cx - 4*u, cy + 3*u, 8*u, 2*u)
        rect(surface, CONTORNO, cx - 4*u, cy + 3*u, 8*u, 2*u, max(1, u//2))

    elif piece_type == chess.KING:
        # Corpo
        pts_corpo = [
            (cx - 3*u, cy + u),
            (cx - 2*u, cy - 3*u),
            (cx + 2*u, cy - 3*u),
            (cx + 3*u, cy + u),
        ]
        poly(surface, CORPO,    pts_corpo)
        poly(surface, CONTORNO, pts_corpo, max(1, u//2))
        # Cruz no topo
        rect(surface, CORPO,    cx - u,       cy - 5*u, 2*u,   4*u)
        rect(surface, CORPO,    cx - 2*u,     cy - 4*u, 4*u,   2*u)
        rect(surface, CONTORNO, cx - u,       cy - 5*u, 2*u,   4*u, max(1, u//2))
        rect(surface, CONTORNO, cx - 2*u,     cy - 4*u, 4*u,   2*u, max(1, u//2))
        # Base
        rect(surface, CORPO,    cx - 3*u, cy + u,   6*u, 2*u)
        rect(surface, CONTORNO, cx - 3*u, cy + u,   6*u, 2*u, max(1, u//2))
        rect(surface, CORPO,    cx - 4*u, cy + 3*u, 8*u, 2*u)
        rect(surface, CONTORNO, cx - 4*u, cy + 3*u, 8*u, 2*u, max(1, u//2))


# ---------------------------------------------------------------------------
# 5. Interface Pygame
# ---------------------------------------------------------------------------

# Paleta do notebook
COR_FUNDO       = (49,  47,  43)
COR_PAINEL      = (241, 242, 216)
COR_CASA_CLARA  = (241, 242, 216)
COR_CASA_ESCURA = (123, 148,  89)
COR_SELECAO     = (200, 210,  80)
COR_LEGAL       = (100, 120, 200)
COR_ULT_LANCE   = (220, 180,  60)
COR_TEXTO_ESC   = (49,  47,  43)
COR_TEXTO_CLA   = (241, 242, 216)
COR_AVISO       = (220,  80,  60)
COR_OK          = (80,  160,  80)

TAMANHO_CASA = 80
BORDA        = 36
PAINEL_W     = 230

LARGURA = BORDA * 2 + TAMANHO_CASA * 8 + PAINEL_W
ALTURA  = BORDA * 2 + TAMANHO_CASA * 8


def _sq_para_pixel(square, flip=False):
    col = chess.square_file(square)
    row = chess.square_rank(square)
    if flip:
        col = 7 - col
        row = 7 - row
    return BORDA + col * TAMANHO_CASA, BORDA + (7 - row) * TAMANHO_CASA


def _pixel_para_sq(px, py, flip=False):
    col = (px - BORDA) // TAMANHO_CASA
    row = (py - BORDA) // TAMANHO_CASA
    if not (0 <= col < 8 and 0 <= row < 8):
        return None
    file = col if not flip else 7 - col
    rank = (7 - row) if not flip else row
    return chess.square(file, rank)


class ChessUI:
    def __init__(self, engine: Engine, jogador_cor: bool = chess.WHITE):
        import pygame
        self.pg = pygame

        self.engine      = engine
        self.jogador_cor = jogador_cor
        self.flip        = (jogador_cor == chess.BLACK)

        self.pg.init()
        self.pg.display.set_caption("Chess Engine — DCP")
        self.screen = self.pg.display.set_mode((LARGURA, ALTURA))
        self.clock  = self.pg.time.Clock()

        # Fonte apenas para texto (não para peças)
        self.font_ui    = self.pg.font.SysFont(None, 18)
        self.font_titulo= self.pg.font.SysFont(None, 22)
        self.font_hist  = self.pg.font.SysFont(None, 16)

        self._resetar()

        # Superfície para overlay translúcido
        self._ov = self.pg.Surface((TAMANHO_CASA, TAMANHO_CASA), self.pg.SRCALPHA)

        # Busca assíncrona em thread separada para não travar o event loop
        self._engine_thread   = None
        self._engine_resultado= None   # lance calculado
        self._engine_rodando  = False

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    def _resetar(self):
        self.board        = chess.Board()
        self.sel_sq       = None
        self.legal_dests  = set()
        self.ult_from     = None
        self.ult_to       = None
        self.historico    = []          # lista de strings SAN
        self.fim_jogo     = False
        self.status_extra = ""          # profundidade atingida, etc.

    # ------------------------------------------------------------------
    # Thread da engine (evita travar UI)
    # ------------------------------------------------------------------

    def _iniciar_busca(self):
        """Dispara a busca em background thread."""
        if self._engine_rodando:
            return
        self._engine_rodando  = True
        self._engine_resultado = None
        board_copia = self.board.copy()

        def _buscar():
            mv = self.engine.escolher_lance(board_copia)
            self._engine_resultado = mv
            self._engine_rodando   = False

        self._engine_thread = threading.Thread(target=_buscar, daemon=True)
        self._engine_thread.start()

    def _aplicar_resultado_engine(self):
        """Chamada no loop principal quando a thread termina."""
        mv = self._engine_resultado
        self._engine_resultado = None

        if mv is None or mv not in self.board.legal_moves:
            return

        san = self.board.san(mv)
        self.ult_from = mv.from_square
        self.ult_to   = mv.to_square
        self.board.push(mv)
        self.historico.append(san)

    # ------------------------------------------------------------------
    # Desenho
    # ------------------------------------------------------------------

    def _overlay(self, x, y, cor_rgb, alpha):
        self._ov.fill((0, 0, 0, 0))
        self.pg.draw.rect(self._ov, (*cor_rgb, alpha), (0, 0, TAMANHO_CASA, TAMANHO_CASA))
        self.screen.blit(self._ov, (x, y))

    def _desenhar_tabuleiro(self):
        for rank in range(8):
            for file in range(8):
                sq  = chess.square(file, rank)
                x, y = _sq_para_pixel(sq, self.flip)
                clara = (file + rank) % 2 == 0
                self.pg.draw.rect(self.screen,
                                  COR_CASA_CLARA if clara else COR_CASA_ESCURA,
                                  (x, y, TAMANHO_CASA, TAMANHO_CASA))

    def _desenhar_destaques(self):
        # Último lance
        if self.ult_from is not None:
            for sq in (self.ult_from, self.ult_to):
                x, y = _sq_para_pixel(sq, self.flip)
                self._overlay(x, y, COR_ULT_LANCE, 100)

        # Seleção atual
        if self.sel_sq is not None:
            x, y = _sq_para_pixel(self.sel_sq, self.flip)
            self._overlay(x, y, COR_SELECAO, 160)

            # Pontos de destino legal
            dot = self.pg.Surface((TAMANHO_CASA, TAMANHO_CASA), self.pg.SRCALPHA)
            for dest in self.legal_dests:
                tx, ty = _sq_para_pixel(dest, self.flip)
                dot.fill((0, 0, 0, 0))
                # Se há peça adversária: anel; se vazio: ponto
                if self.board.piece_at(dest):
                    self.pg.draw.circle(dot, (*COR_LEGAL, 180),
                                        (TAMANHO_CASA//2, TAMANHO_CASA//2),
                                        TAMANHO_CASA//2 - 4, 5)
                else:
                    self.pg.draw.circle(dot, (*COR_LEGAL, 160),
                                        (TAMANHO_CASA//2, TAMANHO_CASA//2), 12)
                self.screen.blit(dot, (tx, ty))

        # Rei em xeque
        if self.board.is_check():
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                x, y = _sq_para_pixel(king_sq, self.flip)
                self._overlay(x, y, COR_AVISO, 140)

    def _desenhar_pecas(self):
        for square, piece in self.board.piece_map().items():
            x, y = _sq_para_pixel(square, self.flip)
            _desenhar_peca_pygame(self.screen, self.pg,
                                  piece.piece_type, piece.color == chess.WHITE,
                                  x, y, TAMANHO_CASA)

    def _desenhar_coords(self):
        for i in range(8):
            file  = i if not self.flip else 7 - i
            letra = chr(ord('a') + file)
            s = self.font_ui.render(letra, True, COR_TEXTO_CLA)
            self.screen.blit(s, (BORDA + i * TAMANHO_CASA + TAMANHO_CASA - s.get_width() - 3,
                                 BORDA + 8 * TAMANHO_CASA - s.get_height() - 2))

            rank = (7 - i) if not self.flip else i
            n = self.font_ui.render(str(rank + 1), True, COR_TEXTO_CLA)
            self.screen.blit(n, (BORDA + 2, BORDA + i * TAMANHO_CASA + 2))

    def _desenhar_painel(self):
        px = BORDA * 2 + TAMANHO_CASA * 8
        self.pg.draw.rect(self.screen, COR_PAINEL, (px, 0, PAINEL_W, ALTURA))
        m = 12   # margem

        def txt(texto, y, cor=COR_TEXTO_ESC, font=None):
            f = font or self.font_ui
            s = f.render(texto, True, cor)
            self.screen.blit(s, (px + m, y))
            return y + s.get_height() + 4

        def linha(y):
            self.pg.draw.line(self.screen, (180, 180, 160),
                              (px + m, y), (px + PAINEL_W - m, y), 1)
            return y + 6

        y = 14
        y = txt("Chess Engine", y, font=self.font_titulo)
        y = txt("DCP  •  CNN Baseline", y, cor=(120, 120, 100))
        y = linha(y)

        # Status
        status = self._status()
        cor_s  = COR_AVISO if self.fim_jogo else COR_TEXTO_ESC
        for linha_s in status.split("\n"):
            y = txt(linha_s, y, cor=cor_s)

        # Indicador de busca
        if self._engine_rodando:
            y = txt("Pensando...", y, cor=(60, 80, 180))

        if self.status_extra:
            y = txt(self.status_extra, y, cor=(80, 120, 80))

        y = linha(y + 2)

        # Controles
        y = txt("[R] reiniciar  [U] desfazer", y, cor=(130, 130, 110))
        y = linha(y + 2)

        # Histórico
        y = txt("Lances", y, cor=(120, 120, 100))
        max_linhas = max(1, (ALTURA - y - 10) // 17)
        pares = []
        for i in range(0, len(self.historico), 2):
            b = self.historico[i]
            p = self.historico[i + 1] if i + 1 < len(self.historico) else ""
            pares.append((b, p))
        for num, (b, p) in enumerate(pares[-max_linhas:], start=max(1, len(pares) - max_linhas + 1)):
            txt(f"{num:2d}. {b:<9} {p}", y, cor=COR_TEXTO_ESC, font=self.font_hist)
            y += 17

    def _status(self):
        b = self.board
        if b.is_checkmate():
            venc = "Pretas" if b.turn == chess.WHITE else "Brancas"
            return f"Xeque-mate!\n{venc} vencem."
        if b.is_stalemate():
            return "Empate - Afogamento"
        if b.is_insufficient_material():
            return "Empate - Mat. insuf."
        if b.is_repetition(3):
            return "Empate - Repeticao"
        if b.can_claim_fifty_moves():
            return "Empate - 50 lances"
        if b.is_check():
            vez = "Brancas" if b.turn == chess.WHITE else "Pretas"
            return f"Xeque! Vez das {vez}"
        vez   = "Brancas" if b.turn == chess.WHITE else "Pretas"
        quem  = "Voce" if b.turn == self.jogador_cor else "Engine"
        return f"Vez das {vez} ({quem})"

    def _renderizar(self):
        self.screen.fill(COR_FUNDO)
        self._desenhar_tabuleiro()
        self._desenhar_destaques()
        self._desenhar_pecas()
        self._desenhar_coords()
        self._desenhar_painel()
        self.pg.display.flip()

    # ------------------------------------------------------------------
    # Lógica humano
    # ------------------------------------------------------------------

    def _selecionar(self, sq):
        peca = self.board.piece_at(sq)
        if peca and peca.color == self.board.turn == self.jogador_cor:
            self.sel_sq      = sq
            self.legal_dests = {mv.to_square for mv in self.board.legal_moves
                                if mv.from_square == sq}
        else:
            self.sel_sq      = None
            self.legal_dests = set()

    def _tentar_mover(self, dest_sq):
        if self.sel_sq is None:
            return

        prom = None
        peca = self.board.piece_at(self.sel_sq)
        if peca and peca.piece_type == chess.PAWN and chess.square_rank(dest_sq) in (0, 7):
            prom = chess.QUEEN   # auto-promove para dama

        mv = chess.Move(self.sel_sq, dest_sq, promotion=prom)
        if mv in self.board.legal_moves:
            san = self.board.san(mv)
            self.ult_from = self.sel_sq
            self.ult_to   = dest_sq
            self.board.push(mv)
            self.historico.append(san)
            self.sel_sq      = None
            self.legal_dests = set()
            return

        # Clicar em outra peça própria → reselecionar
        outra = self.board.piece_at(dest_sq)
        if outra and outra.color == self.jogador_cor:
            self._selecionar(dest_sq)
        else:
            self.sel_sq      = None
            self.legal_dests = set()

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def rodar(self):
        rodando = True
        while rodando:
            self.clock.tick(30)

            # -- Verificar fim de jogo (ANTES de lançar a engine) --
            if not self.fim_jogo and self.board.is_game_over():
                self.fim_jogo = True

            # -- Resultado da engine pronto? --
            if not self._engine_rodando and self._engine_resultado is not None:
                self._aplicar_resultado_engine()

            # -- Lançar busca se for vez da engine e não houver fim --
            if (not self.fim_jogo
                    and not self._engine_rodando
                    and self._engine_resultado is None
                    and self.board.turn != self.jogador_cor):
                self._iniciar_busca()

            self._renderizar()

            # -- Eventos --
            for event in self.pg.event.get():
                if event.type == self.pg.QUIT:
                    rodando = False

                elif event.type == self.pg.KEYDOWN:
                    if event.key == self.pg.K_r and not self._engine_rodando:
                        self._resetar()
                    elif event.key == self.pg.K_u and not self._engine_rodando:
                        # Desfaz 2 meios-lances (lance humano + lance engine)
                        for _ in range(2):
                            if self.board.move_stack:
                                self.board.pop()
                                if self.historico:
                                    self.historico.pop()
                        self.sel_sq      = None
                        self.legal_dests = set()
                        self.ult_from    = None
                        self.ult_to      = None
                        self.fim_jogo    = False
                    elif event.key == self.pg.K_ESCAPE:
                        rodando = False

                elif (event.type == self.pg.MOUSEBUTTONDOWN
                      and event.button == 1
                      and not self.fim_jogo
                      and not self._engine_rodando
                      and self.board.turn == self.jogador_cor):

                    sq = _pixel_para_sq(*event.pos, self.flip)
                    if sq is None:
                        self.sel_sq      = None
                        self.legal_dests = set()
                    elif self.sel_sq is not None:
                        self._tentar_mover(sq)
                    else:
                        self._selecionar(sq)

        self.pg.quit()


# ---------------------------------------------------------------------------
# 6. Ponto de entrada
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chess Engine — DCP")
    parser.add_argument("--modelo", type=str,
                        default="assignment1_outputs/DCP/best_model_baseline_cnn.pt")
    parser.add_argument("--profundidade-min", type=int, default=2,
                        help="Profundidade mínima garantida (default 2)")
    parser.add_argument("--profundidade-max", type=int, default=8,
                        help="Profundidade máxima do iterative deepening (default 8)")
    parser.add_argument("--tempo", type=float, default=4.0,
                        help="Orçamento de tempo por lance em segundos (default 4)")
    parser.add_argument("--cor", choices=["brancas", "pretas"], default="brancas")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo_path = Path(args.modelo)
    if not modelo_path.exists():
        print(f"[AVISO] '{modelo_path}' não encontrado. Usando avaliação de material puro.")
        VALORES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                   chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
        def eval_fn(fen):
            b = chess.Board(fen)
            return float(sum(
                VALORES[p.piece_type] * (1 if p.color == chess.WHITE else -1)
                for p in b.piece_map().values()
            ))
    else:
        print(f"Carregando modelo '{modelo_path}' em {device}...")
        model    = carregar_modelo(modelo_path, device)
        eval_fn  = make_eval_fn(model, device)
        print("Pronto.")

    jogador_cor = chess.WHITE if args.cor == "brancas" else chess.BLACK
    engine_cor  = not jogador_cor

    engine = Engine(
        eval_fn     = eval_fn,
        depth_min   = args.profundidade_min,
        depth_max   = args.profundidade_max,
        time_limit  = args.tempo,
        color       = engine_cor,
    )

    print("\n=== Controles ===")
    print("  Clique  — selecionar / mover")
    print("  R       — reiniciar")
    print("  U       — desfazer último par de lances")
    print("  ESC     — sair\n")

    ui = ChessUI(engine=engine, jogador_cor=jogador_cor)
    ui.rodar()


if __name__ == "__main__":
    main()
