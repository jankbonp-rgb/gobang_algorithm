"""第二级策略评估:候选动作的底片联合签名 σ 与联合特征(个数/步数/动态难度)。

动机(用户规格):场面上多个候选点时,不同的位置牵连的可匹配底片不一样,
威胁值不同。对候选动作 m:
- 触及哪些底片(推进我方的 / 拆对方的),每个底片的步数 k;
- 个数、步数、**动态难度**:
      d(f) = k(f)                          # 还需步数
           + 拦截压力(f)                   # 对方活底片的杀死格落在 f 轨迹上的个数
           + 争点压力(f)                   # 对方底片轨迹与 f 轨迹争抢同一点的个数
  (难度上限 16;含义 = 该底片在实战里被对方拆掉/拖延的容易程度);
- 成对联合(叠加/双威胁雏形/竞速)与一子两用。

σ(m) = 触及底片集合的**结构量化**(不写死关系分类、不用角色/模式类):
- 距离成型步数:双方 live/potential 的 k=1/2/3+ 计数(0/1/2+ 量化);
- 进攻点位数量:双方触及底片的轨迹点总数(分桶量化);
- 与其他底片的叠加关系:同方共享轨迹点(除 m 外)、双威胁雏形(轨迹不相交);
- 与敌方底片的关系:竞速符号、敌杀我、我杀敌(计数量化)。
权重全部未知、由训练学 —— 补丁表的键空间,补丁入口建库即存在。
"""
from __future__ import annotations

from .board import Board, opponent
from .relations import film_instances

DIFF_CAP = 16

JOINT_FEATURE_NAMES: list[str] = [
    # 0-15   触及计数:我方/对方 × live/potential × k=1..4
    "j_my_live_k1", "j_my_live_k2", "j_my_live_k3", "j_my_live_k4",
    "j_my_pot_k1", "j_my_pot_k2", "j_my_pot_k3", "j_my_pot_k4",
    "j_their_live_k1", "j_their_live_k2", "j_their_live_k3", "j_their_live_k4",
    "j_their_pot_k1", "j_their_pot_k2", "j_their_pot_k3", "j_their_pot_k4",
    # 16-17  动态难度加权(Σ1/d,双方;步数影响由上方 per-k 计数自配权重,不写死)
    "j_my_diff_w", "j_their_diff_w",
    # 18-19  最快底片步数(无则 99)
    "j_my_min_k", "j_their_min_k",
    # 20-21  一子两用
    "j_dual_block",     # m 推进我方轨迹,同时落在对方活底片的杀死格上(拆他威胁)
    "j_dual_contend",   # m 推进我方轨迹,同时是双方轨迹争点
    # 22-23  双威胁雏形(m 同时触及两个 k<=2 且轨迹不相交的同方底片)
    "j_my_double", "j_their_double",
    # 24-25  竞速符号(m 触及的异方对里谁更快)
    "j_race_my_fast", "j_race_them_fast",
    # 26-27  触及总数
    "j_my_touched", "j_their_touched",
]


def joint_feature_names() -> list[str]:
    return list(JOINT_FEATURE_NAMES)


def film_touches(board: Board, m, insts) -> list:
    """m 触及的全部底片实例,返回 [(FilmInst, role)];role = 'sq' 轨迹点 / 'kill' 杀死格。"""
    out = []
    for f in insts:
        if m in f.squares:
            out.append((f, "sq"))
        elif f.kill and m in f.kill:
            out.append((f, "kill"))
    return out


def _q(x: int) -> int:
    """计数量化:0 / 1 / 2+(签名用,粗粒度保覆盖)。"""
    return 0 if x == 0 else (1 if x == 1 else 2)


def signature(board: Board, me: int, m, insts) -> tuple:
    """σ(m):触及底片集合的结构量化(不写死关系分类,权重未知、训练学)。

    组成(共 22 项,全部 0/1/2+ 量化,me 视角):
    - 我方 live/potential 的 k=1,k=2,k>=3 计数(6),对方同(6);
    - 双方进攻点位数量(触及底片轨迹点总数,0/1-2/3-5/6+ 分桶)(2);
    - 叠加关系:同方共享轨迹点(除 m 外)、双威胁雏形(2×2=4);
    - 敌方关系:竞速我快/敌快、敌杀我、我杀敌(4)。
    """
    opp = opponent(me)
    touched = film_touches(board, m, insts)
    mine = [f for f, _r in touched if f.owner == me]
    theirs = [f for f, _r in touched if f.owner == opp]

    parts = []
    for side in (mine, theirs):
        live_k1 = live_k2 = live_k3 = 0
        pot_k1 = pot_k2 = pot_k3 = 0
        for f in side:
            if f.is_live:
                if f.k == 1:
                    live_k1 += 1
                elif f.k == 2:
                    live_k2 += 1
                else:
                    live_k3 += 1
            else:
                if f.k == 1:
                    pot_k1 += 1
                elif f.k == 2:
                    pot_k2 += 1
                else:
                    pot_k3 += 1
        parts.extend(map(_q, (live_k1, live_k2, live_k3, pot_k1, pot_k2, pot_k3)))

    def atk_points(items):
        return sum(len(f.squares) for f in items)

    my_pts, their_pts = atk_points(mine), atk_points(theirs)

    def bucket_pts(x):
        return 0 if x == 0 else (1 if x <= 2 else (2 if x <= 5 else 3))

    parts.append(bucket_pts(my_pts))
    parts.append(bucket_pts(their_pts))

    def share_beyond_m(a, b):
        shared = set(a.squares) & set(b.squares)
        shared.discard(m)
        return bool(shared)

    def count_pairs(items, pred):
        n = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if pred(items[i], items[j]):
                    n += 1
        return n

    my_share = count_pairs(mine, share_beyond_m)
    their_share = count_pairs(theirs, share_beyond_m)
    my_dbl = count_pairs(mine, lambda a, b: a.k <= 2 and b.k <= 2
                         and not (set(a.squares) & set(b.squares)))
    their_dbl = count_pairs(theirs, lambda a, b: a.k <= 2 and b.k <= 2
                            and not (set(a.squares) & set(b.squares)))
    parts.extend(map(_q, (my_share, their_share, my_dbl, their_dbl)))

    my_fast = their_fast = they_kill = i_kill = 0
    for a in mine:
        for b in theirs:
            if a.k < b.k:
                my_fast += 1
            elif b.k < a.k:
                their_fast += 1
            if b.kill and set(b.kill) & set(a.squares):
                they_kill += 1
            if a.kill and set(a.kill) & set(b.squares):
                i_kill += 1
    parts.extend(map(_q, (my_fast, their_fast, they_kill, i_kill)))
    return tuple(parts)


def signature_key(board: Board, me: int, m, insts) -> int:
    """σ 的稳定整数哈希(补丁表 dict 键)。

    必须跨进程稳定:Python 内建 hash 带进程随机盐,训练进程存的键在
    推理进程会不同 —— 用 crc32 对签名的 repr 做确定性哈希。
    """
    import zlib

    return zlib.crc32(repr(signature(board, me, m, insts)).encode("utf-8"))


# ---- 细落子指纹(几何写细版 σ,2025 实测修订;第 2 版:镜像/旋转归一)----
# 细键给每个触及底片带:归属/live/k/角色(sq=0,kill=1)/轴向(4 向)/
# 格在底片轨迹上的排位(沿轴向投影排序) + 邻域 cheb1/cheb2 双方子数。
# 第 1 版轴向有向 → 同一"形"8 个方向 8 个键,槽触发稀(自博整局 0 触发)。
# 第 2 版把键按盘面 D4 对称群(镜像×旋转)归一:同一形任何方向/翻面都
# 折叠成同一个键 —— 槽出场 ×4~8,学一次八方向通用(用户规格:扩散)。
# 键仍跨进程稳定(crc32),零槽=中立;只用于新槽族 geo_patches。
_DIRS4 = ((0, 1), (1, 0), (1, 1), (1, -1))
_DIR_ID = {(0, 1): 0, (1, 0): 1, (1, 1): 2, (1, -1): 3}


def _sym_cells():
    """15x15 盘 D4 对称群(8 个):(名字, 映射函数)。"""
    n = 14

    def h(r, c):
        return (r, n - c)

    def v(r, c):
        return (n - r, c)

    def d1(r, c):
        return (c, r)

    def d2(r, c):
        return (n - c, n - r)

    def r90(r, c):
        return (c, n - r)

    def r180(r, c):
        return (n - r, n - c)

    def r270(r, c):
        return (n - c, r)

    def ident(r, c):
        return (r, c)

    return [("id", ident), ("h", h), ("v", v), ("d1", d1), ("d2", d2),
            ("r90", r90), ("r180", r180), ("r270", r270)]


_SYMS = _sym_cells()


def _sym_tables():
    """每个对称下,4 个轴向 dir 的映射: (新 dir id, 是否反转顺序)。"""
    tables = []
    for _name, sym in _SYMS:
        row = []
        for d in range(4):
            dr, dc = _DIRS4[d]
            a0, a1 = sym(4, 4)
            b0, b1 = sym(4 + dr, 4 + dc)
            wr, wc = b0 - a0, b1 - a1
            hit = None
            for e in range(4):
                er, ec = _DIRS4[e]
                if (wr, wc) == (er, ec):
                    hit = (e, False)
                    break
                if (wr, wc) == (-er, -ec):
                    hit = (e, True)
                    break
            row.append(hit or (d, False))
        tables.append(row)
    return tables


_SYM_TABLES = _sym_tables()


def _line_dir_id(board: Board, idx: int) -> int:
    """线 idx 的遍历方向(4 向之一),按 board 缓存。"""
    cache = board.caches.setdefault("line_dirs", {})
    d = cache.get(idx)
    if d is None:
        line = board.lines()[idx]
        r0, c0 = line[0]
        r1, c1 = line[1]
        d = _DIR_ID[(r1 - r0, c1 - c0)]
        cache[idx] = d
    return d


def _fine_tokens(board: Board, me: int, m, insts):
    """未折叠的底片+邻域 token 列表(含每底片轨迹长,供反转排位)。"""
    touched = film_touches(board, m, insts)
    toks = []
    for f, role in touched:
        side = 0 if f.owner == me else 1
        role_id = 0 if role == "sq" else 1
        did = _line_dir_id(board, f.line_index)
        cells = f.squares if role == "sq" else f.kill
        L = len(cells)
        dr, dc = _DIRS4[did]
        order = sorted(cells, key=lambda c: (dr * c[0] + dc * c[1], c[0],
                                             c[1]))
        try:
            rank = order.index(m)
        except ValueError:
            rank = 0
        toks.append((side, int(f.is_live), int(min(f.k, 4)), did, role_id,
                     min(rank, 15), min(L, 16)))
    me_n1 = me_n2 = op_n1 = op_n2 = 0
    r, c = m
    for rr in range(r - 2, r + 3):
        for cc in range(c - 2, c + 3):
            if not (0 <= rr < board.size and 0 <= cc < board.size):
                continue
            if (rr, cc) == (r, c):
                continue
            v = board.grid[rr][cc]
            if v == 0:
                continue
            d = max(abs(rr - r), abs(cc - c))
            if v == me:
                if d == 1:
                    me_n1 += 1
                else:
                    me_n2 += 1
            else:
                if d == 1:
                    op_n1 += 1
                else:
                    op_n2 += 1
    nb = (min(me_n1, 7), min(me_n2, 7), min(op_n1, 7), min(op_n2, 7))
    return toks, nb


def fine_key(board: Board, me: int, m, insts=None, audit=None) -> int:
    """折叠细落子指纹(第 2 版):D4 镜像/旋转归一,跨进程稳定。

    组件与第 1 版相同,但把 (轴向,排位) 按 8 个对称方向折叠:同一形的
    8 个朝向只产生一个键(取 8 个规范表示的字典序最小者);邻域计数
    在 D4 下天然不变。
    """
    import zlib

    if insts is None:
        from .relations import film_instances

        insts = film_instances(board)
    toks, nb = _fine_tokens(board, me, m, insts)
    best = None
    for si in range(8):
        tab = _SYM_TABLES[si]
        out = []
        for t in toks:
            side, live, k, d, role, rank, L = t
            dn, flip = tab[d]
            r2 = (L - 1 - rank) if flip else rank
            out.append((side, live, k, dn, role, r2, L))
        out.sort()
        rep = repr(out) + "|" + repr(nb)
        if best is None or rep < best:
            best = rep
    if audit is not None:
        audit["toks"] = toks
    return zlib.crc32(best.encode("utf-8"))


def _difficulty(f, opp_live_kills, opp_sq_sets) -> int:
    """动态难度 d(f) = k + 拦截压力 + 争点压力,上限 DIFF_CAP。"""
    fsq = set(f.squares)
    inter = 0
    for kill in opp_live_kills:
        if kill and set(kill) & fsq:
            inter += 1
    cont = 0
    for sq in opp_sq_sets:
        if sq & fsq:
            cont += 1
    return min(f.k + inter + cont, DIFF_CAP)


def joint_features(board: Board, me: int, m, insts) -> list[float]:
    """第二级联合特征向量(len == len(JOINT_FEATURE_NAMES),me 视角)。

    步数影响不写死:per-k 计数全部入特征,权重由训练分配;动态难度
    (已确认口径)以 Σ1/d 一个特征进入,重要性同样由权重决定。
    """
    opp = opponent(me)
    opp_live_kills = [f.kill for f in insts
                      if f.owner == opp and f.is_live and f.kill]
    opp_sq_sets = [set(f.squares) for f in insts if f.owner == opp]

    touched = film_touches(board, m, insts)
    mine = [f for f, _r in touched if f.owner == me]
    theirs = [f for f, _r in touched if f.owner == opp]

    cnt = [0.0] * len(JOINT_FEATURE_NAMES)
    I = {n: i for i, n in enumerate(JOINT_FEATURE_NAMES)}

    def bump(f, side, is_live):
        key = f"j_{side}_{'live' if is_live else 'pot'}_k{f.k}"
        cnt[I[key]] += 1.0

    for f in mine:
        bump(f, "my", f.is_live)
    for f in theirs:
        bump(f, "their", f.is_live)

    def diff_w(items):
        return sum(1.0 / _difficulty(f, opp_live_kills, opp_sq_sets) for f in items)

    cnt[I["j_my_diff_w"]] = diff_w(mine)
    cnt[I["j_their_diff_w"]] = diff_w(theirs)
    cnt[I["j_my_min_k"]] = min((f.k for f in mine), default=99)
    cnt[I["j_their_min_k"]] = min((f.k for f in theirs), default=99)

    # 一子两用(注意:冲四的成五点同时是对方底片的轨迹点与杀死格,
    # 判定以"m ∈ 对方活底片杀死格"为准)
    my_sq = any(role == "sq" for f, role in touched if f.owner == me)
    if my_sq:
        for f, role in touched:
            if f.owner == opp and f.is_live and f.kill and m in f.kill:
                cnt[I["j_dual_block"]] += 1.0
        for f in theirs:
            if m in f.squares:
                cnt[I["j_dual_contend"]] += 1.0

    my_k2 = [f for f in mine if f.k <= 2]
    their_k2 = [f for f in theirs if f.k <= 2]
    for a in range(len(my_k2)):
        for b in range(a + 1, len(my_k2)):
            if not (set(my_k2[a].squares) & set(my_k2[b].squares)):
                cnt[I["j_my_double"]] += 1.0
    for a in range(len(their_k2)):
        for b in range(a + 1, len(their_k2)):
            if not (set(their_k2[a].squares) & set(their_k2[b].squares)):
                cnt[I["j_their_double"]] += 1.0
    for a in mine:
        for b in theirs:
            if a.k < b.k:
                cnt[I["j_race_my_fast"]] += 1.0
            elif b.k < a.k:
                cnt[I["j_race_them_fast"]] += 1.0

    cnt[I["j_my_touched"]] = float(len(mine))
    cnt[I["j_their_touched"]] = float(len(theirs))
    return cnt
