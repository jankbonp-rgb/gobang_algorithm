"""深度 H 的强制前推(VCT / victory-by-continuous-threats)与叶子层统计。

在 retrograde 的单线保证与 fusion 的交点融合之上,把"反四链"推广为完整的
AND/OR 强制序列搜索,修补旧版 four_chain_win 的三个洞:

- 反四嵌套:旧版对"被迫的堵子反造四"一律放弃,这里展开我方再反制
  (挡对方的四同时造自己的四),直到分出胜负;
- 第三处连续冲四偷手数:防守方在任意合法位置的反威胁(含远处第三处)
  都进入 AND 分支;
- 多线融合:双四/四三/双活三由 attack_countdown 的融合值自然覆盖。

语义(a = 我方还需落子手数,对方必应不占预算):

- 我方(进攻方 A)走的是 OR 节点:任意一个能维持强制威胁的落点;
- 对方(防守方 D)走的是 AND 节点:全部"杀死格 + 反威胁"都必须被覆盖;
- 杀死格 = 底片窗口中、被 D 占住即令该底片保证失效或拖慢的空位
  (kill_cells,由单线求解器逐格验证 —— 讨论稿"轨迹被覆盖"判定的精确版);
- 反威胁竞速:节点轮到 D 走时,D 的反威胁 v(总手数,含这一手)满足
  v <= d 才可能抢先(d = A 剩余手数);v > d 时 A 必先到,安全剪掉。
  推导:节点后 A 先手,D 剩 v-1 手;A 的第 d 手与 D 的第 v-1 手比轮次,
  D 先到 <=> v-1 < d <=> v <= d。
- 候选域 = 已有棋子切比雪夫距离 2 内的空点:任何造四/融合威胁的落点
  必与其线内棋子相邻(距离 1),故该域完备;不用 potential_films 的轨迹,
  因为其"跨度内无对方子"的过滤会漏掉两端被挡的冲四候选。
- 转置去重:Zobrist 哈希 + 记忆表,同局面不同次序到达共享求解结果。

dag_stats 是同一 DAG 的"叶子层统计":落首手后展开到 H 层(我方手数口径),
提前结束的必胜/必败节点 1:1 投影到叶子层参与计数 —— 讨论稿"DAG 上
提前结束的节点也要统计映射到最后的叶子结点层"的直译,RL 特征 #4 的原料。
"""
from __future__ import annotations

from functools import lru_cache

from .board import EMPTY, Board, opponent
from .fusion import attack_countdown
from .match import live_films, nearby_empties, win_squares_of
from .retrograde import US, guarantee

ATK, DEF = 0, 1


@lru_cache(maxsize=None)
def kill_cells(pattern: str) -> tuple[int, ...]:
    """pattern 内,被防守方占住即杀死该底片的空位索引(含拖慢)。

    底片语义:guarantee(P, US) = k 表示我方先手 k 手内成五。空位 i 是
    杀死格 <=> guarantee(P[i->B], US) 为 None 或 > k —— 单线求解器对
    每个 i 做的是精确判定,不是启发式。
    """
    base = guarantee(pattern, US)
    if base is None:
        return ()
    out = []
    for i, ch in enumerate(pattern):
        if ch != "E":
            continue
        k2 = guarantee(pattern[:i] + "B" + pattern[i + 1 :], US)
        if k2 is None or k2 > base:
            out.append(i)
    return tuple(out)


class _NodeCap(Exception):
    pass


_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def run_length(board: Board, r: int, c: int, player: int) -> int:
    """假设 (r,c) 落 player 子后,经过该点的最长连续长度(零落子预筛)。

    造四 <=> 某个方向上 run >= 4;造三反威胁 <=> run >= 3(含跳型:两侧
    各数一段连续子,自身算 1)。比"落子 + 重匹配"便宜两个数量级。
    """
    best = 1
    for dr, dc in _DIRS:
        run = 1
        for sign in (1, -1):
            rr, cc = r + sign * dr, c + sign * dc
            while board.inside(rr, cc) and board.grid[rr][cc] == player:
                run += 1
                rr += sign * dr
                cc += sign * dc
        if run > best:
            best = run
    return best


def _touched(board: Board, cell) -> set:
    return set(board.lines_through(cell[0], cell[1]))


def threat_moves(board: Board, attacker: int, restrict=None) -> list:
    """A 的候选落点:(攻击值, 坐标) 升序。

    域 = 已有棋子切比雪夫距离 1 内的空点(造四/造三必与其线内棋子相邻)
    ∪ 对方一手成五点(强制挡点);先按连子数 >=3 粗筛再求攻击值。
    restrict 非 None 时,普通候选限定在 restrict 集合内(强制挡点不受限)——
    用于"区域剪枝后集中算力"的受限 VCT。
    """
    defender = opponent(attacker)
    seen: set = set()
    out = []
    for cell in nearby_empties(board, dist=1):
        if cell in seen:
            continue
        seen.add(cell)
        if restrict is not None and cell not in restrict:
            continue
        if run_length(board, cell[0], cell[1], attacker) < 3:
            continue
        v = attack_countdown(board, attacker, cell)
        out.append((v if v is not None else 99, cell))
    for cell in win_squares_of(live_films(board, defender, 1)):
        if cell in seen:
            continue
        seen.add(cell)
        v = attack_countdown(board, attacker, cell)
        out.append((v if v is not None else 99, cell))
    out.sort()
    return out


def counter_moves(board: Board, attacker: int, d: int) -> set:
    """D 的候选落点:杀死 A 的底片(k <= d)的格 ∪ 竞速合法的反威胁(v <= d)。

    attack_countdown 为 None 但落子后 D 有 k<=3 底片的(如单冲四 v=2、
    单活三 v=3),由落子后底片匹配补上;窗口 9 的单线保证封顶 k=2,
    融合封顶 v=3,故 k<=3 的反威胁枚举是完备的。
    """
    defender = opponent(attacker)
    moves: set = set()
    for f in live_films(board, attacker, min(d, 4)):
        if f.coords is None:
            continue
        for i in kill_cells(f.pattern):
            c = f.coords[i]
            if c is not None and board.get(c[0], c[1]) == EMPTY:
                moves.add(c)
    for cell in nearby_empties(board, dist=1):
        if cell in moves:
            continue
        v = attack_countdown(board, defender, cell)
        if v is not None:
            if v <= d:
                moves.add(cell)
            continue
        if run_length(board, cell[0], cell[1], defender) < 3:
            continue                          # 造不出四/三:零落子剪掉
        board.place(cell[0], cell[1], defender)
        try:
            fs = live_films(board, defender, 3, line_indices=_touched(board, cell))
            if fs and min(f.k for f in fs) + 1 <= d:
                moves.add(cell)
        finally:
            board.undo()
    return moves


class VCT:
    """一次搜索:attacker 是否能在 max_d 手内强制取胜,并给出最少手数。

    budget 为跨调用共享的节点预算 dict(键 "n"):用尽后本调用直接放弃,
    返回 None —— 引擎端以此在整手棋的多次 VCT 调用间摊算力。
    """

    def __init__(self, board: Board, attacker: int, max_d: int,
                 node_cap: int = 30000, memo: dict | None = None,
                 budget: dict | None = None, restrict=None):
        self.board = board
        self.A = attacker
        self.D = opponent(attacker)
        self.max_d = max_d
        self.memo = memo if memo is not None else {}
        self.budget = budget if budget is not None else {"n": node_cap}
        self.restrict = restrict  # 区域剪枝:进攻候选限定集(强制挡点除外)

    def _tick(self) -> None:
        self.budget["n"] -= 1
        if self.budget["n"] < 0:
            raise _NodeCap

    def _immediate(self, player: int):
        return win_squares_of(live_films(self.board, player, 1))

    def _after_four(self, player: int, cell) -> bool:
        """cell 刚落 player 的子:是否形成一手成五/四(强制威胁)。"""
        return bool(win_squares_of(
            live_films(self.board, player, 1, line_indices=_touched(self.board, cell))
        ))

    # ---- 递归 ----

    def atk(self, d: int):
        """A 走,还剩 d 手:最小必胜手数(<= d)或 None。"""
        self._tick()
        key = (self.board.zobrist, self.A, ATK)
        hit = self.memo.get(key)
        if hit is not None and (hit[0] is not None or hit[1] >= d):
            return hit[0]                            # 正结果不随深度变;None 结果在
        result = self._atk(d)                        # 更浅探索过时重估
        self.memo[key] = (result, d)
        return result

    def _atk(self, d: int):
        if d <= 0:
            return None
        if self._immediate(self.A):
            return 1
        best = None
        for v, cell in threat_moves(self.board, self.A, self.restrict):
            r, c = cell
            if v is None and run_length(self.board, r, c, self.A) < 4:
                continue                            # 造不出四且无融合:零落子剪掉
            self.board.place(r, c, self.A)
            try:
                if self.board.is_win_move(r, c, self.A):
                    sub = 0
                elif v is not None and v <= d:
                    sub = self.defn(d - 1)          # 融合威胁(四三/双活三/双四)
                elif self._after_four(self.A, cell):
                    sub = self.defn(d - 1)          # 落子后有四(冲四/活四/双四)
                else:
                    continue
                if sub is not None:
                    cand = 1 + sub
                    if cand <= d and (best is None or cand < best):
                        best = cand
            finally:
                self.board.undo()
        return best

    def defn(self, d: int):
        """D 走,A 还剩 d 手:最小必胜手数(<= d)或 None。"""
        self._tick()
        key = (self.board.zobrist, self.A, DEF)
        hit = self.memo.get(key)
        if hit is not None and (hit[0] is not None or hit[1] >= d):
            return hit[0]
        result = self._defn(d)
        self.memo[key] = (result, d)
        return result

    def _defn(self, d: int):
        if self._immediate(self.D):
            return None                             # D 一手成五,反杀
        if d <= 0:
            return None
        films = live_films(self.board, self.A, min(d, 4))
        if not films:
            return None                             # A 无预算内威胁
        moves = counter_moves(self.board, self.A, d)
        if not moves:
            return min(f.k for f in films)          # D 无杀死格/反威胁:A 照最快底片推进
        worst = None
        for cell in sorted(moves):
            r, c = cell
            self.board.place(r, c, self.D)
            try:
                if self.board.is_win_move(r, c, self.D):
                    return None
                sub = self.atk(d)
            finally:
                self.board.undo()
            if sub is None:
                return None
            if worst is None or sub > worst:
                worst = sub
        return worst

    # ---- 入口 ----

    def solve(self):
        """(最少手数, 首手) 或 None(无强制胜 / 节点预算用尽)。

        迭代加深:从 1 手到 max_d 手逐层找,首个命中层即最少手数,
        必胜局面不必证完最小性(逐层首命中即停),亏损局面由节点预算兜底。
        """
        try:
            if self._immediate(self.A):
                return (1, min(self._immediate(self.A)))
            for d in range(1, self.max_d + 1):
                hit = self._solve_any(d)
                if hit is not None:
                    return hit
            return None
        except _NodeCap:
            return None

    def _solve_any(self, d: int):
        """在 d 手预算内找任意一个必胜首手(外层逐层加深保证最少手数)。"""
        for v, cell in threat_moves(self.board, self.A, self.restrict):
            r, c = cell
            if v is None and run_length(self.board, r, c, self.A) < 4:
                continue
            self.board.place(r, c, self.A)
            try:
                if self.board.is_win_move(r, c, self.A):
                    sub = 0
                elif v is not None and v <= d:
                    sub = self.defn(d - 1)
                elif self._after_four(self.A, cell):
                    sub = self.defn(d - 1)
                else:
                    continue
                if sub is not None:
                    return (1 + sub, cell)
            finally:
                self.board.undo()
        return None


def vct_win(board: Board, attacker: int, max_d: int = 5,
            node_cap: int = 30000, memo: dict | None = None,
            budget: dict | None = None, restrict=None):
    """attacker 的强制胜:(最少手数, 首手) 或 None。restrict 限定进攻候选域。"""
    return VCT(board, attacker, max_d, node_cap, memo, budget, restrict).solve()


def dag_stats(board: Board, attacker: int, first_move: tuple,
              H: int, node_cap: int = 80000, fast: bool = False,
              restrict=None):
    """落 first_move 后展开到 H 层(我方手数口径,含首手)的叶子层统计。

    返回 (ratio, total, wins) 或 None(节点超限)。提前结束的必胜/必败节点
    1:1 投影进叶子层:必胜节点计入赢叶子,必败/未知节点计入非赢叶子。
    非理性应对(既不挡威胁也不造反威胁的走法)按证明引理剪掉 —— 它们
    要么让我方照最快底片推进成五,要么慢于竞速条件。
    restrict 非 None 时,我方(进攻方)候选限在该集合内 —— 沿正向偏好
    路径/目标底片域展开更远更特殊的叶子(超剪枝的路径偏好,用户规格);
    防守方应对(杀死格/反威胁)不受限。正常全盘展开照旧(restrict=None)。
    fast=True:D 层只展开杀死格(跳过反威胁枚举),节点上限大幅收紧,
    供每候选特征提取(几十毫秒级)。
    """
    r, c = first_move
    if board.get(r, c) != EMPTY:
        return None
    board.place(r, c, attacker)
    try:
        if board.is_win_move(r, c, attacker):
            return (1.0, 1, 1)                      # 首手即成五:必胜终结
        memo: dict = {}
        try:
            total, wins = _go(board, attacker, H - 1, DEF, node_cap, memo,
                              fast, restrict)
        except _NodeCap:
            return None
    finally:
        board.undo()
    return (wins / total if total else 0.0, total, wins)


def _go(board: Board, attacker: int, d: int, turn: int,
        cap: int, memo: dict, fast: bool = False, restrict=None,
        collect=None, H: int = 0, prior_fn=None, pacc=None):
    """返回 (叶子数, 赢叶子数);collect=1 → 4 元组(分层胜负);
    collect=2 → 6 元组(…, 胜叶记忆分和, 记分叶数):叶终止时若 prior_fn
    与 pacc(预算 {left}) 允许,用该叶局部形查经验表得先验值计入。
    collect=None/0 路径与原先完全一致。
    """
    if len(memo) > cap:
        raise _NodeCap
    key = (board.zobrist, d, attacker, turn)
    if key in memo:
        return memo[key]
    if collect is not None:
        if d < 0 or d > H:
            d = max(0, min(d, H))
    defender = opponent(attacker)
    if turn == ATK:
        if win_squares_of(live_films(board, attacker, 1)):
            _leaf_prior_add(board, prior_fn, pacc)
            res = (1, 1)                            # 必胜终结,投影为 1 个赢叶子
            if collect is not None:
                wd = [0] * (H + 1)
                wd[d] = 1
                res = (1, 1, tuple(wd), (0,) * (H + 1))
        elif d <= 0:
            res = (1, 0)
            if collect is not None:
                res = (1, 0, (0,) * (H + 1), (0,) * (H + 1))
        else:
            total = wins = 0
            if collect is not None:
                wsum = [0] * (H + 1)
                lsum = [0] * (H + 1)
            for v, cell in threat_moves(board, attacker, restrict=restrict):
                if board.get(cell[0], cell[1]) != EMPTY:
                    continue
                if v is None and run_length(board, cell[0], cell[1], attacker) < 4:
                    continue
                board.place(cell[0], cell[1], attacker)
                try:
                    if v is not None and v <= d or win_squares_of(live_films(
                        board, attacker, 1, line_indices=_touched(board, cell)
                    )):
                        r = _go(board, attacker, d - 1, DEF, cap, memo,
                                fast, restrict, collect, H)
                        if collect is not None:
                            _t, _w, wd, ld = r
                            _add_merge(wsum, lsum, wd, ld)
                            total += _t
                            wins += _w
                        else:
                            total += r[0]
                            wins += r[1]
                finally:
                    board.undo()
            if collect is not None:
                res = ((total, wins, tuple(wsum), tuple(lsum))
                       if total else (1, 0, (0,) * (H + 1), (0,) * (H + 1)))
            else:
                res = (total, wins) if total else (1, 0)
    else:
        if win_squares_of(live_films(board, defender, 1)):
            _leaf_prior_add(board, prior_fn, pacc)
            res = (1, 0)                            # 必败终结,投影为 1 个非赢叶子
            if collect is not None:
                ld = [0] * (H + 1)
                ld[d] = 1
                res = (1, 0, (0,) * (H + 1), tuple(ld))
        elif d <= 0:
            res = (1, 0)
            if collect is not None:
                res = (1, 0, (0,) * (H + 1), (0,) * (H + 1))
        else:
            films = live_films(board, attacker, min(d, 4))
            if not films:
                res = (1, 0)
                if collect is not None:
                    res = (1, 0, (0,) * (H + 1), (0,) * (H + 1))
            else:
                if fast:
                    moves = set()
                    for f in films:
                        if f.coords is None:
                            continue
                        for i in kill_cells(f.pattern):
                            c = f.coords[i]
                            if c is not None and board.get(c[0], c[1]) == EMPTY:
                                moves.add(c)
                else:
                    moves = counter_moves(board, attacker, d)
                if not moves:
                    _leaf_prior_add(board, prior_fn, pacc)
                    res = (1, 1)                    # D 无有效应对:A 照最快底片成五
                    if collect is not None:
                        wd = [0] * (H + 1)
                        wd[d] = 1
                        res = (1, 1, tuple(wd), (0,) * (H + 1))
                else:
                    total = wins = 0
                    if collect is not None:
                        wsum = [0] * (H + 1)
                        lsum = [0] * (H + 1)
                    for cell in sorted(moves):
                        board.place(cell[0], cell[1], defender)
                        try:
                            r = _go(board, attacker, d, ATK, cap, memo,
                                    fast, restrict, collect, H)
                            if collect is not None:
                                _t, _w, wd, ld = r
                                _add_merge(wsum, lsum, wd, ld)
                                total += _t
                                wins += _w
                            else:
                                total += r[0]
                                wins += r[1]
                        finally:
                            board.undo()
                    if collect is not None:
                        res = ((total, wins, tuple(wsum), tuple(lsum))
                               if total else (1, 0, (0,) * (H + 1),
                                              (0,) * (H + 1)))
                    else:
                        res = (total, wins) if total else (1, 0)
    memo[key] = res
    return res


def _leaf_prior_add(board, prior_fn, pacc):
    """胜/败叶终止处:用叶局部形查经验表(认形),累进 pacc。"""
    if prior_fn is None or pacc is None:
        return
    if pacc.get("left", 0) <= 0:
        return
    if not board.moves:
        return
    cell = (board.moves[-1][0], board.moves[-1][1])
    from .leafmem import leaf_shape_signature
    sig = leaf_shape_signature(board, cell)
    v = prior_fn(sig)
    pacc["left"] -= 1
    pacc["sum"] = pacc.get("sum", 0.0) + float(v)
    pacc["n"] = pacc.get("n", 0) + 1


def dag_shape_stats(board: Board, attacker: int, first_move: tuple,
                    H: int, node_cap: int = 6000, fast: bool = False,
                    restrict=None, prior_fn=None, budget: int = 200):
    """带认形的深叶统计:落 first_move 后展开 H 层;胜/败叶终止处累加
    局部形经验先验。返回 (hard_ratio, total, wins, mem) 或 None;
    mem = {'n','sum','mean','left'}。默认路径与 dag_stats 等价。"""
    if prior_fn is None:
        budget = 0
    r, c = first_move
    if board.get(r, c) != EMPTY:
        return None
    board.place(r, c, attacker)
    try:
        if board.is_win_move(r, c, attacker):
            return (1.0, 1, 1, {"n": 0, "sum": 0.0, "mean": 0.5,
                                "left": budget})
        memo: dict = {}
        pacc = {"left": budget}
        try:
            total, wins = _go(board, attacker, H - 1, DEF, node_cap, memo,
                              fast, restrict, None, 0, prior_fn, pacc)
        except _NodeCap:
            return None
    finally:
        board.undo()
    n = pacc.get("n", 0)
    mean = (pacc.get("sum", 0.0) / n) if n else 0.5
    mem = {"n": n, "sum": pacc.get("sum", 0.0), "mean": mean,
           "left": pacc.get("left", budget)}
    return (wins / total if total else 0.0, total, wins, mem)


def _add_merge(wsum, lsum, wd, ld):
    for i in range(len(wsum)):
        wsum[i] += wd[i]
        lsum[i] += ld[i]
    for i in range(len(wsum)):
        wsum[i] += wd[i]
        lsum[i] += ld[i]


def layer_profile(board: Board, attacker: int, H: int = 4,
                  node_cap: int = 2000, fast: bool = False,
                  restrict=None):
    """按深度分层的进攻 profile(不落首手,attacker 先行):

    返回 (total, wins, {depth: [win, loss]}) 或 None(节点超限)。
    depth = 终止时已消耗的攻击方手数(0..H);各层胜负比例不同,分层保留。
    restrict:进攻候选域(注意力线);防守应对不受限。
    """
    memo: dict = {}
    try:
        r = _go(board, attacker, H, ATK, node_cap, memo, fast, restrict,
                collect=True, H=H)
    except _NodeCap:
        return None
    if r is None:
        return None
    total, wins, wd, ld = r
    layers = {}
    for d in range(H + 1):
        if wd[d] or ld[d]:
            layers[d] = [wd[d], ld[d]]
    return (total, wins, layers)
