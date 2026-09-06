"""RL 2.0 核心 v3.2(用户修正整合版)。

- 相对坐标:**两轴规范码 (a≤b),每轴 ≤5(横5+竖5 = 曼哈顿10)**,
  超轴 → far(部分回退:不用"和≤10 任意组合");
- 三视图评分:推理同时含 我方推进 / 敌方威胁 / 敌我重合(第三情况),
  三视图各自一套参数(零=中立);占/堵的"格语义"仍统一(见 spec);
- 双层推理:
  * 前向(注意力层展开):每层集 = 下一批注意力底片,评分=三层分;
  * 逆向(末端前推):一个底片逆推两个前驱底片(双威胁雏形),同样按层
    三层打分;棋子数量最少的末端前推结果 → 分配最多算力(预算反比石子数);
- 重复集合键 → 不重展,进入**解方程**:对重复集做固定点迭代求解节点值,
  并回代父节点(不再只是计访问);
- 每层评分(三视图内):S1 集合键分 + S2 深度分 + S3 层胜比×层权。
"""
from __future__ import annotations

import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gomoku.board import BLACK, WHITE, Board, opponent  # noqa: E402
from gomoku.relations import (  # noqa: E402
    FilmInst,
    film_instances,
    film_values,
)

R = 6               # 邻接(哪些底片算相关,Chebyshev)
AXIS_MAX = 5        # 相对坐标每轴上限(横5+竖5=曼哈顿10)
TOP_INIT = 5
TOP_NEXT = 4
MAX_LEVELS = 8
_FAR = (9, 9)
VIEWS = ("my", "enemy", "shared")


def _dir_of(line_idx: int) -> int:
    n = 15
    if line_idx < n:
        return 0
    if line_idx < 2 * n:
        return 1
    if line_idx < 2 * n + 21:
        return 2
    return 3


def axis_of(f: FilmInst) -> int:
    return 0 if _dir_of(f.line_index) < 2 else 1


def angle_class(a: FilmInst, b: FilmInst) -> int:
    aa, ba = axis_of(a), axis_of(b)
    if aa != ba:
        return 45
    return 0 if _dir_of(a.line_index) == _dir_of(b.line_index) else 90


def _desc(f: FilmInst) -> tuple:
    return (f.owner, int(f.is_live), int(min(f.k, 4)), axis_of(f))


def _cells(f: FilmInst) -> tuple:
    return tuple(sorted(set(tuple(c) for c in f.squares) |
                        set(tuple(c) for c in f.kill)))


def rel_vec(ca, cb):
    """两轴规范相对坐标 (a≤b, 每轴≤AXIS_MAX):横5竖5=曼哈顿10。"""
    best = None
    for (r1, c1) in ca:
        for (r2, c2) in cb:
            dx, dy = abs(r1 - r2), abs(c1 - c2)
            if dx > AXIS_MAX or dy > AXIS_MAX:
                continue
            v = (min(dx, dy), max(dx, dy))
            d = dx + dy
            if best is None or d < best[1]:
                best = (v, d)
    return best[0] if best else _FAR


def pair_code(a: FilmInst, b: FilmInst):
    da, db = sorted((_desc(a), _desc(b)))
    ang = angle_class(a, b)
    off = rel_vec(_cells(a), _cells(b))
    share = 1 if set(_cells(a)) & set(_cells(b)) else 0
    return (da, db, ang, off, share)


def set_key(members) -> int:
    pairs = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            pairs.append(pair_code(members[i], members[j]))
    return zlib.crc32(repr(sorted(pairs)).encode("utf-8"))


def view_of(members, me: int) -> str:
    """三视图(相对 me):我方推进 / 敌方威胁 / 敌我重合(第三情况)。"""
    own = {f.owner for f in members}
    if len(own) == 1:
        return "my" if next(iter(own)) == me else "enemy"
    return "shared"


def cells_of(f) -> set:
    return set(tuple(c) for c in f.squares) | \
        set(tuple(c) for c in f.kill)


def stones_of(f: FilmInst) -> int:
    """底片涉及棋子数:5 窗跨度内己方子数 = 5 - k(live/pot 同口径)。"""
    return max(1, 5 - int(f.k))


def overlap_analysis(films) -> dict:
    cell_films: dict[tuple, list] = {}
    for f in films:
        for c in cells_of(f):
            cell_films.setdefault(c, []).append(f)
    multi = {c for c, fs in cell_films.items() if len(fs) >= 2}
    cross = {c for c in multi
             if len({f.owner for f in cell_films[c]}) == 2}
    return {"multi_cells": len(multi), "cross_cells": len(cross),
            "max_hits": max((len(fs) for fs in cell_films.values()),
                            default=0)}


class _P:
    """三视图参数访问器(family: view → {key: w});零=中立。"""

    def __init__(self, params: dict):
        self.raw = params

    def _fam(self, fam: str, view: str) -> dict:
        return (self.raw.get(fam) or {}).get(view) or {}

    def s1(self, view, key):
        return float(self._fam("gs", view).get(key, 0.0))

    def s2(self, view, depth):
        return float(self._fam("depth", view).get(depth, 0.0))

    def s3(self, view, depth, ratio):
        return float(self._fam("ratio", view).get(depth, 0.0)) * \
            (ratio - 0.5)


@dataclass
class Level:
    depth: int
    films: tuple
    view: str
    key: int
    gkey: int = 0               # 打分键 = 几何键 + near_last 桶 + 父层深
    parent: "Level | None" = None
    value: float = 0.0          # 解方程:节点值(三层分+孩子回代)
    repeat_of: int | None = None  # 若该集此前已存在:指向旧键
    stones: int = 0


@dataclass
class R2Search:
    board: Board
    me: int
    top_init: int = TOP_INIT
    top_next: int = TOP_NEXT
    max_levels: int = MAX_LEVELS
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.p = _P(self.params)
        self.levels: list[Level] = []
        self.by_key: dict[int, Level] = {}
        self.films = film_instances(self.board)
        self._vmap = {id(f): v for v, f in
                      film_values(self.board, self.me,
                                  self.params.get("rel_weights") or {})}
        self.oa = overlap_analysis(self.films)

    # ---- 三层评分(视图内)----
    def score(self, lv: Level, win_ratio: float = 0.5) -> float:
        return self.p.s1(lv.view, lv.gkey or lv.key) + \
            self.p.s2(lv.view, lv.depth) + \
            self.p.s3(lv.view, lv.depth, win_ratio)

    def _near_bucket(self, films) -> int:
        """距上一手远近桶:0=<=2, 1=<=5, 2=远(上下文补丁特征)。"""
        if not self.board.moves:
            return 2
        lr, lc = self.board.moves[-1][0], self.board.moves[-1][1]
        dd = min((max(abs(r - lr), abs(c - lc))
                  for f in films for (r, c) in cells_of(f)), default=99)
        return 0 if dd <= 2 else (1 if dd <= 5 else 2)

    def _gkey(self, key, films, parent) -> int:
        return zlib.crc32(f"{key}|{self._near_bucket(films)}|"
                          f"{parent.depth if parent else -1}"
                          .encode("utf-8"))

    # ---- 末端前推(逆向):一个底片 → 两个前驱底片(双威胁雏形)----
    def end_push_pairs(self, f: FilmInst):
        """逆推 f 的两个前驱:与其共享格且互近的底片对(真少子优先)。"""
        cc = cells_of(f)
        near = [g for g in self.films if g is not f and
                (cc & cells_of(g))]
        out = []
        for i in range(len(near)):
            for j in range(i + 1, len(near)):
                a, b = near[i], near[j]
                if cells_of(a) & cells_of(b):
                    stones = stones_of(f) + stones_of(a) + stones_of(b)
                    out.append((stones, (a, b)))
        out.sort(key=lambda t: t[0])       # 真少子 → 最前(最多算力)
        return out

    def related(self, parent: tuple) -> list:
        pcells = set()
        for f in parent:
            pcells |= cells_of(f)
        out = []
        for f in self.films:
            cc = cells_of(f)
            if not cc:
                continue
            if min(max(abs(r - r2), abs(c - c2))
                   for (r, c) in pcells for (r2, c2) in cc) <= R:
                out.append(f)
        return out

    def _add_level(self, depth, films, parent) -> Level | None:
        key = set_key(films)
        if key in self.by_key:             # 重复集合 → 解方程(见 solve)
            self.by_key[key].repeat_of = key
            self.by_key[key].value += 1.0
            return None
        if depth >= self.max_levels:
            return None
        lv = Level(depth, tuple(films), view_of(films, self.me), key,
                   parent,
                   stones=sum(stones_of(f) for f in films))
        lv.gkey = self._gkey(key, tuple(films), parent)
        self.levels.append(lv)
        self.by_key[key] = lv
        return lv

    # ---- 先验(v0 信号;参数族全 0 时推理仍可工作,训练可叠加覆盖)----
    def prior(self, lv: Level) -> float:
        cells = set()
        cross = multi = 0
        for i, f in enumerate(lv.films):
            fc = cells_of(f)
            cells |= fc
            for g in lv.films[:i]:
                if fc & cells_of(g):
                    multi += 1
                    if f.owner != g.owner:
                        cross += 1
        v = 0.05 * multi + 0.15 * cross
        if lv.view == "shared":
            v += 0.3
        return v + 0.05 / (1 + lv.depth)

    def rev_alloc(self, films) -> int:
        """少子 → 最多递归预算:平均子数越少给的逆向剩余层越多。"""
        avg = sum(stones_of(f) for f in films) / max(1, len(films))
        return int(max(2, min(8, 10 - avg)))

    def run(self):
        """前向+逆向双层推理,逐层三层分+少子预算;末尾固定点解方程。"""
        top = sorted(self.films, key=lambda f: -self._vmap.get(id(f), 0.0))
        init = top[: self.top_init]
        if not init:
            return [], {}
        root = self._add_level(0, init, None)
        if root is None:
            return [], {}
        frontier = [(root, self.rev_alloc(init))]
        while frontier and len(self.levels) < self.max_levels * 2:
            lv, rev_left = frontier.pop(0)
            # --- 前向:下一批注意力底片(域内评分 = 训练分 + 先验 + 交汇)---
            scored = []
            for f in self.related(lv.films):
                bonus = 0.0
                for g in lv.films:
                    if cells_of(f) & cells_of(g):
                        bonus += 2.0
                        if f.owner != g.owner:
                            bonus += 1.0
                cand = Level(lv.depth + 1, (f,), view_of((f,), self.me),
                             set_key((f,)))
                base = self.prior(cand) + self.score(cand) + \
                    self._vmap.get(id(f), 0.0)
                scored.append((base + bonus, f))
            scored.sort(key=lambda t: -t[0])
            chosen = [f for _s, f in scored[: self.top_next]]
            if chosen:
                child = self._add_level(lv.depth + 1, chosen, lv)
                if child:
                    frontier.append((child, self.rev_alloc(chosen)))
            # --- 逆向(末端前推):少子排前,预算内真正向下深扩 ---
            if rev_left <= 0:
                continue
            used = 0
            for f in list(lv.films)[: self.top_next]:
                pairs = self.end_push_pairs(f)
                for stones, (a, b) in pairs:
                    if used >= rev_left:
                        break
                    pair_lv = Level(lv.depth + 1, (a, b),
                                    view_of((a, b), self.me),
                                    set_key((a, b)))
                    pair_lv.value = self.prior(pair_lv) + \
                        self.score(pair_lv) * (2.0 - 1.0 / (1.0 + stones))
                    child = self._add_level(lv.depth + 1, (a, b), lv)
                    if child:
                        child.value = pair_lv.value
                        used += 1
                        # 少子结果继承剩余预算继续深扩(用掉 1 层)
                        nxt = self.rev_alloc((a, b))
                        if nxt > 1:
                            frontier.append((child, min(rev_left - 1,
                                                        nxt - 1)))
                    if used >= rev_left:
                        break
        # ---- 解方程:固定点迭代(节点值 = 先验+三层分 + 孩子均值回代)----
        for _ in range(50):
            delta = 0.0
            for lv in self.levels:
                kids = [c for c in self.levels
                        if c.parent is lv and c.repeat_of is None]
                v = self.prior(lv) + self.score(lv)
                if kids:
                    v += sum(k.value for k in kids) / len(kids)
                d = abs(v - lv.value)
                if d > delta:
                    delta = d
                lv.value = v
            if delta < 1e-3:
                break
        out = []
        for lv in self.levels:
            out.append({"depth": lv.depth, "view": lv.view,
                        "key": lv.key, "gkey": lv.gkey,
                        "n": len(lv.films),
                        "value": round(lv.value, 3),
                        "stones": lv.stones})
        return out, {"levels": len(self.levels),
                     "views": {v: sum(1 for l in self.levels
                                      if l.view == v)
                               for v in VIEWS},
                     "oa": self.oa}
