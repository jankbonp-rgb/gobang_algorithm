"""棋盘与线段。v1 规则假设:自由规则(freestyle),任意方向连成 >=5 即胜,无禁手。"""
from __future__ import annotations

import random

EMPTY, BLACK, WHITE = 0, 1, 2

# 四个方向落在 D4 对称的两条轨道上:{横,竖} 与 {两条对角}。
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

_ZOB_RNG = random.Random(0x517CC1B727220A95)
_ZOB_TABLES: dict = {}


def _zob_table(size: int):
    """尺寸对应的 Zobrist 异或表(进程内共享,确定性)。"""
    table = _ZOB_TABLES.get(size)
    if table is None:
        table = [
            [[_ZOB_RNG.getrandbits(64) for _ in range(3)] for _ in range(size)]
            for _ in range(size)
        ]
        _ZOB_TABLES[size] = table
    return table


def opponent(player: int) -> int:
    return BLACK + WHITE - player


class Board:
    def __init__(self, size: int = 15):
        self.size = size
        self.grid = [[EMPTY] * size for _ in range(size)]
        self.moves: list[tuple[int, int, int]] = []
        self._lines = self._build_lines()
        self._lines_through: dict[tuple[int, int], list[int]] = {}
        for idx, line in enumerate(self._lines):
            for cell in line:
                self._lines_through.setdefault(cell, []).append(idx)
        # 字典化匹配的基础:每条线一个版本号,落子/悔子只令经过的 <=4 条线失效
        self.line_versions = [0] * len(self._lines)
        self.caches: dict = {}
        self._code_cache: list = [None] * len(self._lines)
        # 增量 Zobrist:place/undo 各异或一次,用于 VCT 的转置去重
        self._zob = 0
        self._zob_t = _zob_table(size)

    @property
    def zobrist(self) -> int:
        return self._zob

    def line_code(self, idx: int) -> str:
        """线内容的紧凑编码(内容寻址用):试落+悔子后内容复原即命中。"""
        hit = self._code_cache[idx]
        ver = self.line_versions[idx]
        if hit is not None and hit[0] == ver:
            return hit[1]
        code = "".join(str(self.grid[r][c]) for (r, c) in self._lines[idx])
        self._code_cache[idx] = (ver, code)
        return code

    def _build_lines(self) -> list[list[tuple[int, int]]]:
        n = self.size
        lines: list[list[tuple[int, int]]] = []
        for r in range(n):
            lines.append([(r, c) for c in range(n)])
        for c in range(n):
            lines.append([(r, c) for r in range(n)])
        for d in range(-(n - 1), n):
            diag = [(r, r - d) for r in range(n) if 0 <= r - d < n]
            if len(diag) >= 5:
                lines.append(diag)
        for s in range(2 * n - 1):
            anti = [(r, s - r) for r in range(n) if 0 <= s - r < n]
            if len(anti) >= 5:
                lines.append(anti)
        return lines

    def lines(self) -> list[list[tuple[int, int]]]:
        return self._lines

    def lines_through(self, r: int, c: int) -> list[int]:
        return self._lines_through[(r, c)]

    def inside(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def get(self, r: int, c: int) -> int:
        return self.grid[r][c]

    def place(self, r: int, c: int, player: int) -> None:
        if self.grid[r][c] != EMPTY:
            raise ValueError(f"cell ({r},{c}) is not empty")
        self.grid[r][c] = player
        self.moves.append((r, c, player))
        self._zob ^= self._zob_t[r][c][player]
        for idx in self._lines_through[(r, c)]:
            self.line_versions[idx] += 1

    def undo(self) -> None:
        r, c, p = self.moves.pop()
        self.grid[r][c] = EMPTY
        self._zob ^= self._zob_t[r][c][p]
        for idx in self._lines_through[(r, c)]:
            self.line_versions[idx] += 1

    def empties(self) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.grid[r][c] == EMPTY
        ]

    def full(self) -> bool:
        return len(self.moves) == self.size * self.size

    def is_win_move(self, r: int, c: int, player: int) -> bool:
        """(r,c) 处为 player 的子(已落或假设落下)时是否构成 >=5 连。"""
        for dr, dc in DIRECTIONS:
            count = 1
            for sign in (1, -1):
                rr, cc = r + sign * dr, c + sign * dc
                while self.inside(rr, cc) and self.grid[rr][cc] == player:
                    count += 1
                    rr += sign * dr
                    cc += sign * dc
            if count >= 5:
                return True
        return False
