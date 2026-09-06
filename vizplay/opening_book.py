"""开局冠军模仿表查询(黑/白对称)。

- 训练(experiments/opening_imitate.py)以黑1 为原点、|dx|,|dy|<=3 域内
  摆好局面问 Rapfi 冠军下一步,存 opening_book_<side>.json;
- 本模块:给定实际局面(轮到 side),折叠前缀 → 查表 → 命中的话把规范
  坐标冠军着按当前前缀的对称逆变换还原为实际落点;
- 域外 / 键未命中 / 落点被占 / 出界 → None(调用方走临时思考);
- 前缀折叠 = D4 旋转镜像 + 带色排序,与训练器同一份代码,保证键一致。
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_BOOKS: dict[str, dict] = {}
_ORIGIN = (7, 7)          # 默认:黑1 在盘心(训练即此);实际以首手黑1 为准
_W = 3

_F = [lambda p: (p[0], p[1]),
      lambda p: (-p[0], p[1]),
      lambda p: (p[0], -p[1]),
      lambda p: (-p[0], -p[1]),
      lambda p: (p[1], p[0]),
      lambda p: (-p[1], p[0]),
      lambda p: (p[1], -p[0]),
      lambda p: (-p[1], -p[0])]
_FINV = [lambda p: (p[0], p[1]),
         lambda p: (-p[0], p[1]),
         lambda p: (p[0], -p[1]),
         lambda p: (-p[0], -p[1]),
         lambda p: (p[1], p[0]),
         lambda p: (p[1], -p[0]),
         lambda p: (-p[1], p[0]),
         lambda p: (-p[1], -p[0])]


def _load(side_name: str) -> dict:
    if side_name in _BOOKS:
        return _BOOKS[side_name]
    path = ROOT / "external" / f"opening_book_{side_name}.json"
    d = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for r in data.get("rows", []):
            d[r["key"]] = r
    except Exception:
        pass
    _BOOKS[side_name] = d
    return d


def fold_with(cells):
    """前缀(相对带色 [(r,c,p)...]) D4 规范折叠 → (规范序列, 对称索引)。"""
    best = None
    best_i = 0
    for i, f in enumerate(_F):
        out = tuple(sorted((f((r, c))[0], f((r, c))[1], p)
                           for (r, c, p) in cells))
        if best is None or out < best:
            best, best_i = out, i
    return best, best_i


def book_move(board, side: int):
    """轮到 side 的开局表查询;命中返回实际落点 (r, c),否则 None。"""
    from gomoku.board import BLACK

    if side != BLACK:
        # 白侧表尚未生成时直接 None(对称扩展后启用)
        if not (ROOT / "external" / "opening_book_white.json").exists():
            return None
    name = "black" if side == 1 else "white"
    book = _load(name)
    if not book:
        return None
    moves = board.moves
    if len(moves) < 1 or len(moves) >= 20:
        return None
    if moves[0][2] != BLACK:
        return None                    # 训练假设黑1 先手
    o = (moves[0][0], moves[0][1])
    cells = [(m[0] - o[0], m[1] - o[1], m[2]) for m in moves]
    if any(abs(r) > _W or abs(c) > _W for (r, c, _p) in cells):
        return None                    # 出域 → 临时思考
    seq, i = fold_with(cells)
    key = zlib.crc32(repr(seq).encode("utf-8"))
    row = book.get(key)
    if not row or row.get("out_of_domain"):
        return None
    spec = tuple(row["mv"])
    mv = _FINV[i](spec)
    r, c = o[0] + mv[0], o[1] + mv[1]
    if not (0 <= r < board.size and 0 <= c < board.size):
        return None
    if board.get(r, c) != 0:
        return None
    return (r, c)
