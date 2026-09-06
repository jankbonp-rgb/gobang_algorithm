"""底片关系 + 区域聚类。

底片关系 = 手工版 NNUE 特征:把"双方活底片/潜在底片"之间的关系(竞速、
争点、共线、交叉、拆威胁、双威胁)压成一个固定长度的实值码本,供策略层
在区域上直接打分;区域剪枝后集中算力 —— 先按底片锚点做切比雪夫膨胀再取
8 邻域连通分量,把整盘问题切成若干局部区域,每个区域内的底片数 F 通常
很小(<< 20),于是 O(F^2) 的两两关系扫描也足够便宜。

本模块只读不写:它调用 match.live_films / match.potential_films / vct.kill_cells
(三者自带缓存),绝不修改棋盘。
"""
from __future__ import annotations

from dataclasses import dataclass

from .board import BLACK, EMPTY, WHITE, Board, opponent
from .language import shape_signature
from .match import live_films, potential_films, win_squares_of
from .vct import kill_cells


@dataclass
class FilmInst:
    """底片实例(己/敌、live/potential 统一)。"""

    owner: int          # 该底片属于谁(BLACK/WHITE)
    k: int              # 倒计时
    them_proof: bool    # 仅 live 有意义;potential 恒 False
    squares: tuple      # 轨迹点坐标(live=第一步轨迹;potential=跨度内空点)
    kill: tuple         # 杀死格坐标(live=kill_cells 映射 coords 后坐标集合;potential=空)
    line_index: int
    is_live: bool
    cls: str = ""       # 模式类:live=形状签名;potential="P{k}"(第二级 σ 签名用)


@dataclass
class Region:
    cells: list         # 区域内盘内空格(sorted)
    films: list         # FilmInst 列表
    centroid: tuple     # 锚点重心 ((round(mean_r), round(mean_c)))

    def n_cells(self) -> int:
        return len(self.cells)


def _live_kill(board: Board, f) -> tuple:
    """live 底片的杀死格:kill_cells(pattern) 的索引映射 coords,跳过 None 与界外。"""
    coords = f.coords
    if coords is None:
        return ()
    out = set()
    for i in kill_cells(f.pattern):
        c = coords[i]
        if c is None:
            continue
        r, cc = c
        if board.inside(r, cc):
            out.add((r, cc))
    return tuple(sorted(out))


def film_instances(board: Board, max_k: int = 4) -> list[FilmInst]:
    """双方全部底片实例 = 双方 live_films(max_k) + 双方 potential_films(max_k)。

    live 的 kill 用 vct.kill_cells(f.pattern) 的索引映射 f.coords(注意 coords 可能 None,
    跳过 None 与界外);potential 的 kill 为空 tuple。注意同一线同一跨度的 live 与
    potential 会重复出现,不必去重。
    """
    out: list[FilmInst] = []
    for player in (BLACK, WHITE):
        for f in live_films(board, player, max_k):
            out.append(
                FilmInst(
                    owner=player,
                    k=f.k,
                    them_proof=f.them_proof,
                    squares=f.squares,
                    kill=_live_kill(board, f),
                    line_index=f.line_index,
                    is_live=True,
                    cls=shape_signature(f.pattern),
                )
            )
        for p in potential_films(board, player, max_k):
            out.append(
                FilmInst(
                    owner=player,
                    k=p.k,
                    them_proof=False,
                    squares=p.squares,
                    kill=(),
                    line_index=p.line_index,
                    is_live=False,
                    cls=f"P{p.k}",
                )
            )
    return out


_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _ray_total(board: Board, r: int, c: int) -> int:
    """过 (r,c) 的 8 射线(半径 4)内的双方棋子总数(数量特征热点判据)。"""
    total = 0
    for dr, dc in _DIRS:
        for sign in (1, -1):
            rr, cc = r + sign * dr, c + sign * dc
            for _ in range(4):
                if not board.inside(rr, cc):
                    break
                if board.grid[rr][cc] != EMPTY:
                    total += 1
                rr += sign * dr
                cc += sign * dc
    return total


def cluster_regions(board: Board, max_k: int = 4, radius: int = 2) -> list[Region]:
    """区域聚类(用户规格):几个区域 = 强匹配底片 + 数量特征热点 + 对方最新落子邻域。

    种子三源:
    1. 匹配度大的:双方 live 底片中 k<=2 者,其轨迹点与杀死格;
    2. 数量特征明显的:8 射线双方棋子总数 >=4 的盘内格(密度热点);
    3. 敌人最近下的:最后一手的切比雪夫 radius 邻域。
    种子做 radius 膨胀后取 8 邻域连通分量;棋盘无棋子/无种子时返回空。
    """
    insts = film_instances(board, max_k)
    n = board.size
    seeds: set = set()

    # 1. 强匹配底片(k<=2 的 live)
    for f in insts:
        if f.is_live and f.k <= 2:
            seeds.update(f.squares)
            seeds.update(f.kill)

    # 2. 数量特征热点
    for r in range(n):
        for c in range(n):
            if _ray_total(board, r, c) >= 4:
                seeds.add((r, c))

    # 3. 对方最新落子的邻域
    if board.moves:
        lr, lc, _p = board.moves[-1]
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = lr + dr, lc + dc
                if 0 <= rr < n and 0 <= cc < n:
                    seeds.add((rr, cc))

    if not seeds:
        return []

    # 切比雪夫膨胀
    dilated: set = set()
    for (r, c) in seeds:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    dilated.add((rr, cc))

    # 8 邻域连通分量
    components: list[list] = []
    seen: set = set()
    for cell in sorted(dilated):
        if cell in seen:
            continue
        comp: list = []
        stack = [cell]
        seen.add(cell)
        while stack:
            (r, c) = stack.pop()
            comp.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = (r + dr, c + dc)
                    if nb in dilated and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        components.append(comp)

    regions: list[Region] = []
    for comp in components:
        cells = sorted((r, c) for (r, c) in comp if board.get(r, c) == EMPTY)
        anc = [(r, c) for (r, c) in comp if (r, c) in seeds]
        if anc:
            mr = round(sum(r for r, _ in anc) / len(anc))
            mc = round(sum(c for _, c in anc) / len(anc))
        else:
            mr = mc = 0
        regions.append(Region(cells, [], (mr, mc)))

    # 底片归区:一个实例只归一个区域 —— 相交时归 cells 数最多者,平手归序号小者
    cell_sets = [set(r.cells) for r in regions]
    films_by_region: list[list] = [[] for _ in regions]
    for f in insts:
        sq = set(f.squares)
        best_i = -1
        best_n = -1
        for i, cs in enumerate(cell_sets):
            if sq & cs:
                if len(regions[i].cells) > best_n:
                    best_n = len(regions[i].cells)
                    best_i = i
        if best_i >= 0:
            films_by_region[best_i].append(f)
    for i, r in enumerate(regions):
        r.films = films_by_region[i]
    return regions


REL_FEATURE_NAMES: list[str] = [
    # 0-3   我方按 k=1..4 计数
    "my_k1", "my_k2", "my_k3", "my_k4",
    # 4-7   对方按 k=1..4 计数
    "their_k1", "their_k2", "their_k3", "their_k4",
    # 8-11  live/potential 计数
    "my_live", "my_pot", "their_live", "their_pot",
    # 12-14 同方对(我-我)
    "my_pair_same_line", "my_pair_share_sq", "my_pair_min_k_le2",
    # 15-17 同方对(敌-敌)
    "their_pair_same_line", "their_pair_share_sq", "their_pair_min_k_le2",
    # 18-27 异方对(我-敌)
    "race_them_fast2", "race_them_fast1", "race_even", "race_my_fast1", "race_my_fast2",
    "they_kill_me", "i_kill_them", "share_sq", "same_line", "cross",
    # 28-29 双威胁
    "double_threat_my", "double_threat_their",
    # 30-32 区域标量
    "n_cells", "n_films", "centroid_dist",
]

_IDX = {name: i for i, name in enumerate(REL_FEATURE_NAMES)}


# ---- 底片交互值模型(用户规格)----
# V_i = Σ_j W[rel(i,j), k_i桶, k_j桶]:每个底片的值 = 其他底片对它的贡献之和;
# 可分离几何增量(坐标信息只隐含在槽位里):距离 ≤4 内最近格位移量化为 15 个
# 规范化偏移槽 {rel}_off:{id}(先验 0);dbl 只对近域对触发(跨场双威胁排除),
# cross 叠加位移增量(交叉对齐完备);kill 格交判据;race 纯步数语义。
RACE_BUCKETS = {-2: "race:m2", -1: "race:m1", 0: "race:0", 1: "race:p1", 2: "race:p2"}

PRIORS: dict = {
    "they_kill_me": -0.2,   # 对方杀死格压住我方轨迹(危险)
    "i_kill_them": 0.2,     # 我方杀死格压住对方轨迹(强度)
    "dbl": 0.2,             # 同方双威胁雏形
    "race:m2": -0.4, "race:m1": -0.2, "race:0": 0.0,
    "race:p1": 0.2, "race:p2": 0.4,
    "share": 0.0, "same_line": 0.0, "cross": 0.0,
}


def _kb(k: int) -> int:
    return min(k, 3)


def rel_key(rel: str, ka: int, kb: int) -> str:
    return f"{rel}:{_kb(ka)}:{_kb(kb)}"


# ---- 几何增量助手(可分离二维相对坐标打分,用户规格)----
# 编码原则:键只隐含坐标信息 —— 模型不存绝对坐标;成对现算两底片最近格
# 的位移 (dr,dc),量化成 Chebyshev ≤4 内的 15 个规范化偏移(|dr|,|dc| 排序,
# 转置不变)。增量槽 {rel}_off:{id} 先验 0.0,与 {rel}:{k_i}:{k_j} 基值解耦。
OFF_GRID = [(a, b) for b in range(5) for a in range(b + 1)]  # 15 个偏移
_OFF_ID = {k: i for i, k in enumerate(OFF_GRID)}


def nearest_offset(ca, cb):
    """两格集最近格的规范化位移 (a,b)(0<=a<=b);空集或距离 >4 → None。

    距离 = 切比雪夫;>4 视为独立战场(五连窗口外),不产生几何增量。"""
    if not ca or not cb:
        return None
    best = None
    for (r1, c1) in ca:
        for (r2, c2) in cb:
            d = max(abs(r1 - r2), abs(c1 - c2))
            if d > 4:
                continue
            key = (min(abs(r1 - r2), abs(c1 - c2)),
                   max(abs(r1 - r2), abs(c1 - c2)))
            if best is None or d < best[1]:
                best = (key, d)
    return best[0] if best else None


def film_values(board: Board, me: int, rel_weights=None, audit=None) -> list:
    """底片交互值:返回 [(V, FilmInst)] 按 V 降序(双方全部底片)。

    rel_weights: {槽位键: float} 学习权重;缺失槽位用 PRIORS 先验。
    键两类,可分离:
    - 基值槽 {rel}:{k_i}:{k_j} —— 竞速/杀格/共享/同线的步数语义;
    - 几何增量槽 {rel}_off:{id} —— 距离 ≤4 内两底片最近格的 15 规范化
      位移(坐标信息只隐含在槽位里),先验 0.0。
    dbl(同方 k<=2 双威胁)只对 ≤4 的近域对触发 —— 跨场双威胁结构性排除;
    cross 触发时叠加位移增量(交叉对齐完备)。race 纯步数不加几何。
    audit 非 None 时,向其写入 {id(f): [(槽位键, 该槽先验值), ...]}。
    复杂度 O(n^2),n = 底片数;几何成对现算,不入库。
    """
    w = dict(rel_weights) if rel_weights else {}

    def base(rel, ka, kb):
        """基值槽 {rel}:{k_i}:{k_j},先验 = PRIORS[rel](缺失槽位)。"""
        key = rel_key(rel, ka, kb)
        prior = float(PRIORS.get(rel, 0.0))
        return w.get(key, prior), key, prior

    def geo_inc(rel, near):
        """几何增量槽 {rel}_off:{id},先验 0.0(near = 规范化位移或 None)。"""
        if near is None:
            return 0.0, None
        key = f"{rel}_off:{_OFF_ID[near]}"
        return w.get(key, 0.0), key

    def Wrace(delta):
        k = RACE_BUCKETS.get(max(-2, min(2, delta)), "race:0")
        return w.get(k, PRIORS.get(k, 0.0)), k, PRIORS.get(k, 0.0)

    insts = film_instances(board)
    opp = opponent(me)
    mine = [f for f in insts if f.owner == me]
    theirs = [f for f in insts if f.owner == opp]
    vals = {id(f): 0.0 for f in insts}

    def add(f, x, meta=None):
        vals[id(f)] += x
        if audit is not None and meta is not None:
            audit.setdefault(id(f), []).append(meta)

    # 同方对(对称贡献):基值槽(竞速语义)+ 几何增量槽(位移隐含坐标)。
    # dbl 只对最近格 Chebyshev ≤4 的近域对触发(跨场双威胁结构性排除)。
    for group in (mine, theirs):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                near = nearest_offset(a.squares, b.squares)
                if a.line_index == b.line_index:
                    v, k, p = base("same_line", a.k, b.k)
                    add(a, v, (k, p))
                    add(b, v, (k, p))
                if set(a.squares) & set(b.squares):
                    v, k, p = base("share", a.k, b.k)
                    add(a, v, (k, p))
                    add(b, v, (k, p))
                if _cross(board, a, b):
                    v, k, p = base("cross", a.k, b.k)
                    add(a, v, (k, p))
                    add(b, v, (k, p))
                    if near is not None:          # 交叉对齐:叠加位移增量
                        inc, ok = geo_inc("cross", near)
                        if ok:
                            add(a, inc, (ok, 0.0))
                            add(b, inc, (ok, 0.0))
                if (a.k <= 2 and b.k <= 2 and near is not None
                        and not (set(a.squares) & set(b.squares))):
                    v, k, p = base("dbl", a.k, b.k)
                    add(a, v, (k, p))
                    add(b, v, (k, p))
                    inc, ok = geo_inc("dbl", near)
                    if ok:
                        add(a, inc, (ok, 0.0))
                        add(b, inc, (ok, 0.0))

    # 异方对(有向:从各自视角收贡献)。kill 判据 = 杀格与轨迹格相交(贴邻压迫);
    # share/same_line/cross(叠加位移增量)同同方对。race 恒加。
    for a in mine:
        for b in theirs:
            asq = set(a.squares)
            bsq = set(b.squares)
            near = nearest_offset(a.squares, b.squares)
            # 我方视角
            if b.kill and set(b.kill) & asq:
                v, k, p = base("they_kill_me", b.k, a.k)
                add(a, v, (k, p))
            if a.kill and set(a.kill) & bsq:
                v, k, p = base("i_kill_them", a.k, b.k)
                add(a, v, (k, p))
            v, k, p = Wrace(b.k - a.k)
            add(a, v, (k, p))
            # 对方视角(镜像)
            if a.kill and set(a.kill) & bsq:
                v, k, p = base("they_kill_me", a.k, b.k)
                add(b, v, (k, p))
            if b.kill and set(b.kill) & asq:
                v, k, p = base("i_kill_them", b.k, a.k)
                add(b, v, (k, p))
            v, k, p = Wrace(a.k - b.k)
            add(b, v, (k, p))
            if asq & bsq:
                v, k, p = base("share", a.k, b.k)
                add(a, v, (k, p))
                add(b, v, (k, p))
            if a.line_index == b.line_index:
                v, k, p = base("same_line", a.k, b.k)
                add(a, v, (k, p))
                add(b, v, (k, p))
            if _cross(board, a, b):
                v, k, p = base("cross", a.k, b.k)
                add(a, v, (k, p))
                add(b, v, (k, p))
                if near is not None:
                    inc, ok = geo_inc("cross", near)
                    if ok:
                        add(a, inc, (ok, 0.0))
                        add(b, inc, (ok, 0.0))

    out = [(vals[id(f)], f) for f in insts]
    out.sort(key=lambda t: -t[0])
    return out


def film_key(f: FilmInst) -> tuple:
    """FilmInst 的跨调用稳定内容键:同局面两次枚举的同一底片键相同。

    id() 是对象身份(每次 film_instances 调用都新建对象),不可跨调用比较;
    内容键 = (归属, 线号, live, k, 模式类, 轨迹, 杀死格) 用于审计/标签对齐。
    """
    return (f.owner, f.line_index, f.is_live, f.k, f.cls, f.squares, f.kill)


def champion_films(board: Board, move: tuple, max_k: int = 4) -> list:
    """冠军落点的底片标签(用户规格:超剪枝的训练信号)。

    冠军(蒸馏教师)在当前局势下的落子 move 触碰到的底片,即训练时"当前
    贡献应指向"的底片正集:move ∈ 底片轨迹点 ∪ 杀死格;若一个都不碰
    (纯位置手),退化为 move 切比雪夫 2 邻域内有轨迹/杀死格的底片。
    """
    insts = film_instances(board, max_k)
    direct = [f for f in insts
              if move in set(f.squares) or move in set(f.kill)]
    if direct:
        return direct
    r, c = move
    near = []
    for f in insts:
        cells = set(f.squares) | set(f.kill)
        if any(max(abs(x - r), abs(y - c)) <= 2 for (x, y) in cells):
            near.append(f)
    return near


def directed_domain(board: Board, me: int, rel_weights=None,
                    margin: float = 0.5, cap: int = 5, radius: int = 2):
    """超剪枝(正向路径偏好)的聚焦域(用户规格)。

    超剪枝是正向推理的偏好:在当前匹配底片的全部信息下,决定"将来往哪条
    路径/目标底片发展",而非逆向分析本身 —— VCT 才是逆向(从叶子反推)。
    实现:重算 n 个底片的交互贡献值 V;只对最重要
    的前几个底片做分析 —— 取 V 与最高值差 margin 内且非负的至多 cap 个
    底片(K 不写死),其轨迹点与杀死格做切比雪夫 radius 膨胀作为分析域;
    对方一手成五点(强制挡点)永不受剪。正常逆向分析照旧另行执行。
    返回 (top_films, domain);无底片/无正值时返回 ([], set())。
    """
    values = film_values(board, me, rel_weights)
    if not values:
        return [], set()
    best_v = values[0][0]
    top = []
    for v, f in values:
        if v < best_v - margin:
            break
        if v < 0.0:
            continue
        top.append(f)
        if len(top) >= cap:
            break
    if not top:
        return [], set()
    seeds = set()
    for f in top:
        seeds.update(f.squares)
        seeds.update(f.kill)
    domain = set()
    for (r, c) in seeds:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < board.size and 0 <= cc < board.size:
                    domain.add((rr, cc))
    # 对方一手成五的强制挡点:永不剪(必胜>不输 硬闸门不受超剪枝影响)
    domain.update(win_squares_of(live_films(board, opponent(me), 1)))
    return top, domain


def relation_feature_names() -> list[str]:
    return list(REL_FEATURE_NAMES)


def _unordered_pairs(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


def _share_sq(a, b) -> bool:
    return bool(set(a.squares) & set(b.squares))


def _hit(kill: tuple, squares: tuple) -> bool:
    """kill(可能为空)与 squares 是否有交。"""
    return bool(kill) and bool(set(kill) & set(squares))


def _cross(board: Board, a, b) -> bool:
    """两条线不同,且 a.squares、b.squares 中存在 (x, y) 使 x,y 在同一线号上。"""
    if a.line_index == b.line_index:
        return False
    for (r1, c1) in a.squares:
        la = board.lines_through(r1, c1)
        for (r2, c2) in b.squares:
            lb = board.lines_through(r2, c2)
            if set(la) & set(lb):
                return True
    return False


def region_features(board: Board, me: int, region: Region) -> list[float]:
    """区域关系特征向量(len == len(REL_FEATURE_NAMES),以走子方 me 为视角)。"""
    opp = opponent(me)
    mine = [f for f in region.films if f.owner == me]
    theirs = [f for f in region.films if f.owner == opp]
    I = _IDX
    cnt = [0] * len(REL_FEATURE_NAMES)

    # k 计数
    for f in mine:
        cnt[I["my_k%d" % f.k]] += 1
    for f in theirs:
        cnt[I["their_k%d" % f.k]] += 1

    # live / potential
    cnt[I["my_live"]] = sum(1 for f in mine if f.is_live)
    cnt[I["my_pot"]] = sum(1 for f in mine if not f.is_live)
    cnt[I["their_live"]] = sum(1 for f in theirs if f.is_live)
    cnt[I["their_pot"]] = sum(1 for f in theirs if not f.is_live)

    # 同方对(我-我)
    for a, b in _unordered_pairs(mine):
        if a.line_index == b.line_index:
            cnt[I["my_pair_same_line"]] += 1
        shared = _share_sq(a, b)
        if shared:
            cnt[I["my_pair_share_sq"]] += 1
        if a.k <= 2 and b.k <= 2:
            cnt[I["my_pair_min_k_le2"]] += 1
            if not shared:
                cnt[I["double_threat_my"]] += 1

    # 同方对(敌-敌)
    for a, b in _unordered_pairs(theirs):
        if a.line_index == b.line_index:
            cnt[I["their_pair_same_line"]] += 1
        shared = _share_sq(a, b)
        if shared:
            cnt[I["their_pair_share_sq"]] += 1
        if a.k <= 2 and b.k <= 2:
            cnt[I["their_pair_min_k_le2"]] += 1
            if not shared:
                cnt[I["double_threat_their"]] += 1

    # 异方对(我-敌)
    for m in mine:
        for t in theirs:
            d = t.k - m.k
            if d <= -2:
                cnt[I["race_them_fast2"]] += 1
            elif d == -1:
                cnt[I["race_them_fast1"]] += 1
            elif d == 0:
                cnt[I["race_even"]] += 1
            elif d == 1:
                cnt[I["race_my_fast1"]] += 1
            else:
                cnt[I["race_my_fast2"]] += 1
            if _hit(t.kill, m.squares):
                cnt[I["they_kill_me"]] += 1
            if _hit(m.kill, t.squares):
                cnt[I["i_kill_them"]] += 1
            if _share_sq(m, t):
                cnt[I["share_sq"]] += 1
            if m.line_index == t.line_index:
                cnt[I["same_line"]] += 1
            if _cross(board, m, t):
                cnt[I["cross"]] += 1

    # 区域标量
    cnt[I["n_cells"]] = region.n_cells()
    cnt[I["n_films"]] = len(region.films)
    cr, cc = region.centroid
    center = board.size // 2
    cnt[I["centroid_dist"]] = abs(cr - center) + abs(cc - center)

    return [float(x) for x in cnt]
