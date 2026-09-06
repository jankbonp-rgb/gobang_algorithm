"""前端匹配:把当前棋盘匹配到底片库,得到双方的活底片与倒计时。"""
from __future__ import annotations

from dataclasses import dataclass

from .board import EMPTY, Board, opponent
from .language import board_windows
from .retrograde import THEM, US, Film, best_first_moves, guarantee


def immediate_win_squares(board: Board, player: int) -> set[tuple[int, int]]:
    """一手成五的点(倒计时 0 的入口)。"""
    out = set()
    for (r, c) in board.empties():
        if board.is_win_move(r, c, player):
            out.add((r, c))
    return out


def _line_films(
    board: Board, idx: int, player: int, max_k: int, window: int
) -> list[Film]:
    films = []
    for _idx, pattern, coords in board_windows(board, player, (idx,), window):
        k = guarantee(pattern, US)
        if k is None or k == 0 or k > max_k:
            continue
        them_proof = guarantee(pattern, THEM) is not None
        squares = tuple(
            coords[i] for i in best_first_moves(pattern) if coords[i] is not None
        )
        if not squares:
            continue
        films.append(Film(pattern, k, them_proof, squares, idx, tuple(coords)))
    return films


def live_films(
    board: Board, player: int, max_k: int = 4, line_indices=None, window: int = 9
) -> list[Film]:
    """匹配 player 的活底片(倒计时 <= max_k,即预期层以内)。

    line_indices 非 None 时只重扫这些线(落子后的增量路径,线必脏,不缓存);
    全盘路径走"线版本字典":每条线的匹配结果按版本号缓存,一手棋只令
    经过的 <=4 条线失效,其余线直接命中 —— 这就是字典化的前端匹配。
    """
    if line_indices is not None:
        films = []
        for idx in line_indices:
            films.extend(_line_films(board, idx, player, max_k, window))
        return films
    cache = board.caches.setdefault("films", {})
    out: list[Film] = []
    for idx in range(len(board.lines())):
        key = (idx, player, max_k, window)
        ver = board.line_versions[idx]
        hit = cache.get(key)
        if hit is None or hit[0] != ver:
            hit = (ver, _line_films(board, idx, player, max_k, window))
            cache[key] = hit
        out.extend(hit[1])
    return out


def win_squares_of(films: list[Film]) -> set[tuple[int, int]]:
    """倒计时为 1 的底片的落点 = 一手成五点。"""
    return {sq for f in films if f.k == 1 for sq in f.squares}


@dataclass(frozen=True)
class PotentialFilm:
    """潜在底片:一条五格跨度,已显影 5-k 枚我方子,轨迹(k 个空位)全空。

    这是讨论稿"出现底片且轨迹为空即进入倒计时"的直译 —— 不要求局部保证;
    保证语义(guarantee)仍然只用于 必胜/不输 的判定。k = 剩余步数(层号)。
    """

    k: int
    squares: tuple
    line_index: int


def potential_films(board: Board, player: int, max_k: int = 4) -> list[PotentialFilm]:
    """player 的全部潜在底片(1 <= k <= max_k,即预期层以内)。线版本字典缓存。"""
    opp = opponent(player)
    cache = board.caches.setdefault("pot", {})
    out: list[PotentialFilm] = []
    for idx, line in enumerate(board.lines()):
        key = (idx, player, max_k)
        ver = board.line_versions[idx]
        hit = cache.get(key)
        if hit is None or hit[0] != ver:
            films = []
            vals = [board.get(r, c) for (r, c) in line]
            for i in range(len(line) - 4):
                span = vals[i : i + 5]
                if opp in span:
                    continue
                empties = [line[i + j] for j in range(5) if span[j] == EMPTY]
                k = len(empties)
                if 1 <= k <= min(max_k, 4):  # k=5 纯空跨度未成底片,k=0 已成五
                    films.append(PotentialFilm(k, tuple(empties), idx))
            hit = (ver, films)
            cache[key] = hit
        out.extend(hit[1])
    return out


def nearby_empties(board: Board, dist: int = 2) -> list[tuple[int, int]]:
    """已有棋子切比雪夫距离 dist 内的空点;空盘时取天元。"""
    if not board.moves:
        c = board.size // 2
        return [(c, c)]
    near = set()
    for (r, c, _p) in board.moves:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                rr, cc = r + dr, c + dc
                if board.inside(rr, cc) and board.get(rr, cc) == 0:
                    near.add((rr, cc))
    return sorted(near)
