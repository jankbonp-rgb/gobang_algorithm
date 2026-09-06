"""M2 特征层:候选点生成 + 六类特征源的特征向量提取。

六类来源(用户规格):
1. 区域密度 —— 过候选点 8 射线窗口内己/敌子数;
2. 数量差 —— 密度差与密度和(差值丢失"双方都密"的对杀信息,故和也入特征);
3. 底片匹配度 —— 己方活底片轨迹命中(越近层越重)、对方活底片杀死格命中;
4. DAG 叶子胜率 —— 净进攻分支 + 最快成五 + 一维叶子胜率(浅层代理;
   `vct.dag_stats` 的深前推统计留作离线精算,训练与推理同构即可);
5. 必赢/不输 —— 硬闸门,不进特征(由 rl_agent 复用 frontier 的闸门);
6. 对手最新落子的底片匹配 —— 经过对方最后一手的线段上的底片统计:
   己方活底片数、对方活底片数、己方轨迹点命中、对方杀死格命中、切比雪夫距离。

所有特征以"当前走子方"为视角,颜色对称(蒸馏数据两色共用)。
"""
from __future__ import annotations

from collections import defaultdict

from .board import DIRECTIONS, EMPTY, BLACK, Board, WHITE, opponent
from .filmsig import JOINT_FEATURE_NAMES, joint_features
from .fusion import attack_countdown
from .match import live_films, nearby_empties, potential_films
from .relations import film_instances
from .vct import kill_cells

FEATURE_NAMES = [
    "den_me",        # 0  8 射线己方密度
    "den_opp",       # 1  8 射线对方密度
    "den_diff",      # 2  密度差
    "den_sum",       # 3  密度和
    "my_attack",     # 4  落子后我方的进攻倒计时(None->99,越小越强)
    "fastest",       # 5  最快成五手数(含后续分支,None->99)
    "breadth",       # 6  我方下一层进攻分支数(<=3 的点数)
    "their_breadth", # 7  对方同类分支数
    "net_breadth",   # 8  净分支(我方-对方)
    "my_hit",        # 9  我方活底片轨迹命中(按 1/k 加权)
    "opp_hit",       # 10 对方活底片杀死格命中(挡其威胁)
    "lm_my",         # 11 对方最后一手所在线段上的我方活底片数
    "lm_opp",        # 12 对方最后一手所在线段上的对方活底片数
    "lm_my_hit",     # 13 对方最后一手所在线段上、我方潜在轨迹 == 候选点
    "lm_opp_kill",   # 14 对方最后一手所在线段上、对方杀死格 == 候选点
    "lm_dist",       # 15 候选点距对方最后一手的切比雪夫距离(0..4 截断)
    "center",        # 16 中心偏好(-曼哈顿距离)
    "phase",         # 17 局面阶段(手数/盘面)
    "ratio",         # 18 一维同族叶子胜率(leaf_stats,偏乐观,仅作特征)
    "dag_ratio",     # 19 DAG 叶子层胜率:落该手后 H 层 AND/OR 展开,
                     #    提前结束节点 1:1 投影到叶子层(vct.dag_stats,讨论稿特征 #4)
]

BIG = 99

# 点级特征 = 原始 20 维 + 第二级底片联合信息(filmsig,个数/步数/动态难度)
POINT_FEATURE_NAMES = FEATURE_NAMES + JOINT_FEATURE_NAMES


def dag_ratio_feature(board: Board, me: int, m, H: int = 2,
                      node_cap: int = 400) -> float:
    """特征 #4:落 m 后的 DAG 叶子层赢节点占比(0..1,节点超限取 0)。

    快速统计模式:D 层只展开杀死格,几十毫秒级,供逐候选提取。"""
    from .vct import dag_stats

    s = dag_stats(board, me, m, H, node_cap, fast=True)
    return 0.0 if s is None else s[0]


def candidate_cells(board: Board) -> list:
    """候选点 = 双方潜在底片轨迹并集(与 frontier 同域);空则棋子邻域。"""
    cells: set = set()
    for p in (BLACK, WHITE):
        for f in potential_films(board, p, 4):
            cells.update(f.squares)
    if not cells:
        cells = set(nearby_empties(board))
    return sorted(c for c in cells if board.get(c[0], c[1]) == EMPTY)


def _ray_density(board: Board, m, me: int) -> tuple[int, int]:
    """过 m 的 8 射线(半径 4)内 (己方子数, 对方子数)。"""
    dm = do = 0
    r0, c0 = m
    for dr, dc in DIRECTIONS:
        for sign in (1, -1):
            rr, cc = r0 + sign * dr, c0 + sign * dc
            for _ in range(4):
                if not board.inside(rr, cc):
                    break
                v = board.grid[rr][cc]
                if v == me:
                    dm += 1
                elif v != EMPTY:
                    do += 1
                rr += sign * dr
                cc += sign * dc
    return dm, do


def extract_features(board: Board, me: int, m,
                     my_films, their_films, push, insts=None) -> list:
    """候选点 m 的特征向量(me 视角,点级 = 20 维原始 + 30 维底片联合)。

    push = frontier._push 的返回值元组;insts = relations.film_instances(board)
    的双方底片实例(每局面算一次传入,避免逐候选重算)。
    """
    opp = opponent(me)
    dm, do = _ray_density(board, m, me)
    feats = [dm, do, dm - do, dm + do]

    my_attack = push[4]
    fastest = push[7]
    breadth = push[6]
    their_breadth = push[3]
    ratio = push[8]
    feats += [
        BIG if my_attack is None else my_attack,
        BIG if fastest is None else fastest,
        breadth,
        their_breadth,
        breadth - their_breadth,
    ]

    my_hit = sum(1.0 / f.k for f in my_films if m in f.squares)
    opp_hit = 0.0
    for f in their_films:
        if f.coords is None:
            continue
        killers = kill_cells(f.pattern)
        if any(f.coords[i] == m for i in killers):
            opp_hit += 1.0 / f.k
    feats += [my_hit, opp_hit]

    last = board.moves[-1] if board.moves else None
    if last is not None:
        lr, lc, _lp = last
        lines = set(board.lines_through(lr, lc))
        lm_my = sum(1 for f in my_films if f.line_index in lines)
        lm_opp = sum(1 for f in their_films if f.line_index in lines)
        lm_my_hit = sum(
            1 for f in potential_films(board, me, 4)
            if f.line_index in lines and m in f.squares
        )
        lm_opp_kill = 0
        for f in their_films:
            if f.line_index not in lines or f.coords is None:
                continue
            killers = kill_cells(f.pattern)
            if any(f.coords[i] == m for i in killers):
                lm_opp_kill += 1
        lm_dist = min(max(abs(m[0] - lr), abs(m[1] - lc)), 4)
    else:
        lm_my = lm_opp = lm_my_hit = lm_opp_kill = lm_dist = 0
    feats += [lm_my, lm_opp, lm_my_hit, lm_opp_kill, lm_dist]

    c = board.size // 2
    feats.append(-(abs(m[0] - c) + abs(m[1] - c)))
    feats.append(len(board.moves) / (board.size * board.size))
    feats.append(ratio)
    feats.append(dag_ratio_feature(board, me, m))  # 特征 #4:DAG 叶子层胜率
    if insts is None:
        insts = film_instances(board)
    feats.extend(joint_features(board, me, m, insts))  # 第二级:个数/步数/动态难度
    return feats


def merged_feature_names() -> list:
    """点级特征(20 原始 + 30 底片联合) + 区域级底片关系特征 的完整名字表。"""
    from .relations import REL_FEATURE_NAMES

    return POINT_FEATURE_NAMES + list(REL_FEATURE_NAMES)


def merged_features(point_feats, region_feats) -> list:
    """点级特征与区域关系特征拼接为平坦向量。"""
    return [float(x) for x in point_feats] + [float(x) for x in region_feats]
