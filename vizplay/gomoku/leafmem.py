"""叶子记忆:局面几何指纹 → 经验分(递归评估的基例)。

签名:对局面(me 视角)取双方 top 结构 —— 全部 live k<=2 底片与 potential
k<=3 底片的 (owner,k,live,cls,线方向桶,跨度形态) 排序元组 CRC;绝对坐标
不入签名(平移不变,叶子可出现在任意区域)。
经验:主库样本中 mover 侧 == 终局胜者 → 1 否则 0(和局跳过),按签名聚合。
用途:推理叶子的伪模拟记忆分(训练时组合分),出现次数 >= MIN_N 才可用。
"""
from __future__ import annotations

import zlib

from .relations import film_instances

_DIR_ID = {(0, 1): 0, (1, 1): 1, (1, 0): 2, (-1, 1): 3}


def _dir_bucket(board, li: int) -> int:
    lines = board.lines()
    if not (0 <= li < len(lines)) or len(lines[li]) < 2:
        return 0
    a, b = lines[li][0], lines[li][1]
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dc < 0 or (dc == 0 and dr < 0):
        dr, dc = -dr, -dc
    return _DIR_ID.get((dr, dc), 0)


def geometry_signature(board, me, max_films: int = 12) -> int:
    """局面几何指纹:top 底片 (owner,k,live,cls,方向,span尺寸) 排序 CRC。"""
    insts = film_instances(board)
    sigs = []
    for f in insts:
        if not f.is_live and f.k > 3:
            continue                      # potential k=4 单子跨度:噪声
        cells = f.squares or f.kill
        span = len(cells)
        d = _dir_bucket(board, f.line_index)
        sigs.append((f.owner, min(f.k, 4), int(f.is_live), f.cls, d,
                     min(span, 6)))
    sigs.sort()
    sigs = sigs[:max_films]
    return zlib.crc32(repr(sigs).encode("utf-8"))


def _rel_sig(insts, cell, board, me_side):
    """与 cell 相关的局部底片形条目:坐标→相对 cell 的位移桶(平移不变)。"""
    out = []
    r, c = cell
    for f in insts:
        cells = set(f.squares) | set(f.kill)
        if not cells:
            continue
        best = None
        for (r2, c2) in cells:
            dd = max(abs(r2 - r), abs(c2 - c))
            if best is None or dd < best:
                best = dd
        if best is not None and best <= 4:
            d = _dir_bucket(board, f.line_index)
            out.append((f.owner, min(f.k, 4), int(f.is_live), f.cls, d,
                        min(best, 4)))
    out.sort()
    return tuple(out)


def leaf_shape_signature(board, cell, max_insts: int = 10) -> int:
    """叶子局部形指纹:最后一手 cell 半径 4 内的底片
    (owner,k,live,cls,方向,距 cell 桶) 排序 CRC —— 认形用(平移不变)。"""
    insts = film_instances(board)
    sigs = []
    r, c = cell
    for f in insts:
        if not f.is_live and f.k > 3:
            continue
        cells = set(f.squares) | set(f.kill)
        if not cells:
            continue
        best = min((max(abs(r2 - r), abs(c2 - c)) for (r2, c2) in cells),
                   default=99)
        if best <= 4:
            d = _dir_bucket(board, f.line_index)
            sigs.append((f.owner, min(f.k, 4), int(f.is_live), f.cls, d,
                         best))
    sigs.sort()
    sigs = sigs[:max_insts]
    return zlib.crc32(repr(sigs).encode("utf-8"))
