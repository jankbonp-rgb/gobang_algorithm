"""棋盘底层语言 v1:一维窗口语言。

字母表(以"我方"视角):
    M = 我方子   E = 空(可能属于某条底片的轨迹)   B = 不可用(对方子或界外)

设计依据:四个成五方向在 D4 对称下只有两条轨道(轴向/45 度),压到一维后
两类共享同一套线上语言;二维性只在多线交点组合(双四、四三)时出现,
那是语言的下一层,不在 v1 的一维库里。
"""
from __future__ import annotations

from .board import EMPTY, Board, opponent

WINDOW = 9  # 5(成五) + 两侧各 2 的语境,足以区分活/眠与跳型
M, E, B = "M", "E", "B"
_HALF = WINDOW // 2


def canonical(s: str) -> str:
    """一维窗口的规范形:正读与反读取字典序小者(镜像对称)。"""
    r = s[::-1]
    return s if s <= r else r


def shape_signature(s: str) -> str:
    """形状签名:去掉两端 B、内部 B 连段并为一个 B,再取规范形。
    用来度量"底层语言把多少原始状态压成同一个结构"。"""
    core = s.strip(B)
    collapsed = []
    for ch in core:
        if ch == B and collapsed and collapsed[-1] == B:
            continue
        collapsed.append(ch)
    return canonical("".join(collapsed))


def board_windows(board: Board, player: int, line_indices=None, window: int = WINDOW):
    """以 player 视角,把每条线切成长度 window 的窗口(界外补 B)。

    产出 (line_index, pattern, coords):coords 与 pattern 逐位对齐,补位处为 None。
    每个真实格子恰好居中出现一次。line_indices 非 None 时只扫这些线(增量匹配)。
    window 就是"算力档位":9/11/13 对应末端前推的三档状态空间。
    """
    lines = board.lines()
    indices = range(len(lines)) if line_indices is None else line_indices
    half = window // 2
    for idx in indices:
        line = lines[idx]
        chars = []
        for (r, c) in line:
            v = board.get(r, c)
            chars.append(M if v == player else (E if v == EMPTY else B))
        padded = B * half + "".join(chars) + B * half
        coords: list = [None] * half + list(line) + [None] * half
        for i in range(len(line)):
            yield idx, padded[i : i + window], coords[i : i + window]
