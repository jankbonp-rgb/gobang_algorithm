"""交点组合层:混合结构 = 两条线在交点上的合法融合(位置无关)。

对应讨论中的三类前推底片与剪枝:
1. 结构内部的强制前推 —— retrograde.guarantee 已覆盖(单线 OR/AND 求解);
2. 等价结构路由 —— 步数<=的结构从不临时重建:单线值全部走规范形记忆表,
   融合值由两条线的库值 O(1) 合成,库条目是"引用对",不物化任何位置;
3. 混合结构 —— 两条不平行的线恰好只交于一格,融合的合法性约束只有
   "交点状态一致"一条,因此位置无关、不同摆放不单独看待。

融合定理(在交点落我子后,轮对方走):
    若两条线各自的先手保证 kA、kB 都存在,则融合倒计时 = 1 + max(kA, kB),
    且对方先手也挡不住 —— 对方每手只能干预一条线(两线仅共享已落的交点),
    另一条线的保证照常推进;对方下在别处对单线语义等价于弃权,已被单线
    求解器覆盖。双冲四 = 1+max(1,1) = 2,四三 = 1+max(1,2) = 3,双活三 = 3。
已知边界:对方在第三处用连续冲四偷手数不在此保证内(全局反四链,待做)。
"""
from __future__ import annotations

from .board import DIRECTIONS, EMPTY, Board, opponent
from .match import live_films, potential_films, win_squares_of
from .retrograde import THEM, US, guarantee


def centered_patterns(board: Board, player: int, cell, window: int = 9) -> list[str]:
    """经过 cell 的四个方向、以 cell 为中心的窗口(player 视角),直接提取。"""
    half = window // 2
    r, c = cell
    grid = board.grid
    out = []
    for dr, dc in DIRECTIONS:
        chars = []
        for off in range(-half, half + 1):
            rr, cc = r + dr * off, c + dc * off
            if 0 <= rr < board.size and 0 <= cc < board.size:
                v = grid[rr][cc]
                chars.append("M" if v == player else ("E" if v == EMPTY else "B"))
            else:
                chars.append("B")
        out.append("".join(chars))
    return out


def attack_value_at(board: Board, player: int, cell, max_k: int = 3) -> int | None:
    """cell 处已有 player 的子时,该子激活的进攻倒计时(含该子,对方先手)。

    取两种来源的最小值:单线对方先手也挡不住(如活四);
    两条线的融合(双四/四三/双三,用融合定理合成)。无保证返回 None。
    """
    best = None
    us_ks = []
    for p in centered_patterns(board, player, cell):
        kt = guarantee(p, THEM)
        if kt is not None and kt <= max_k:
            best = min(best, 1 + kt) if best is not None else 1 + kt
        ku = guarantee(p, US)
        if ku is not None and ku <= max_k:
            us_ks.append(ku)
    if len(us_ks) >= 2:
        us_ks.sort()
        fused = 1 + us_ks[1]  # 最快两条线:1 + max(两个最小 k)
        best = min(best, fused) if best is not None else fused
    return best


def four_chain_win(board: Board, attacker: int, budget: int):
    """反四链(连续冲四)的强制取胜:全首端推理的"推到静止"部分。

    单局面字典装不下它的原因:链里对方的应手不是"局部防守/弃权",
    而是被我方的四强制 —— 攻防方向逐手反转,只能沿强制序列推进。
    语义(保守健全,只认无歧义的强制):attacker 每手必须成四(或成五);
    防守方仅当没有自己的一手成五时才被迫堵唯一成五点;若被迫的堵子
    反造出防守方的四,该分支视为被反杀而放弃。
    返回 (attacker 总手数, 第一手) 或 None。budget = attacker 手数上限。
    """
    defender = opponent(attacker)
    now = win_squares_of(live_films(board, attacker, 1))
    if now:
        return (1, min(now))
    if budget <= 1:
        return None
    cells = sorted(
        {sq for f in potential_films(board, attacker) if f.k == 2 for sq in f.squares}
    )
    best = None
    for cell in cells:
        board.place(cell[0], cell[1], attacker)
        try:
            comps = win_squares_of(live_films(board, attacker, 1))
            if not comps:
                continue
            if len(comps) >= 2:
                if win_squares_of(live_films(board, defender, 1)):
                    continue  # 防守方反有一手成五
                cand = (2, cell)  # 双成五点:堵一个,下另一个
                if best is None or cand < best:
                    best = cand
                continue
            if win_squares_of(live_films(board, defender, 1)):
                continue  # 防守方无视我方的四直接成五
            comp = next(iter(comps))
            board.place(comp[0], comp[1], defender)
            try:
                if win_squares_of(live_films(board, defender, 1)):
                    continue  # 被迫的堵子反造四(偷回手数),保守放弃
                sub = four_chain_win(board, attacker, budget - 1)
                if sub is not None:
                    cand = (1 + sub[0], cell)
                    if best is None or cand < best:
                        best = cand
            finally:
                board.undo()
        finally:
            board.undo()
    return best


def attack_countdown(board: Board, player: int, cell, max_k: int = 3) -> int | None:
    """在空格 cell 落 player 子的进攻倒计时(含这一手,对方先手)。

    字典加速:值只取决于经过 cell 的 <=4 条线的内容,以
    (cell, player, 各线版本号) 为键缓存 —— 远离最近落点的格子直接命中,
    不必重复推理。
    """
    if board.get(cell[0], cell[1]) != EMPTY:
        return None
    key = (
        cell[0], cell[1], player, max_k,
        tuple(board.line_code(i) for i in board.lines_through(*cell)),
    )
    cache = board.caches.setdefault("attack", {})
    hit = cache.get(key, -1)
    if hit != -1:
        return hit
    board.place(cell[0], cell[1], player)
    try:
        value = attack_value_at(board, player, cell, max_k)
    finally:
        board.undo()
    if len(cache) > 200_000:
        cache.clear()
    cache[key] = value
    return value
