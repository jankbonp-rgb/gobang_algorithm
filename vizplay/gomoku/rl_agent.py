"""M2 策略引擎(两级):区域剪枝 → 集中算力深推理 → 平坦权重评分。

分层(与讨论稿一致):
- 必赢:一手成五 / 融合 <=2 / 强制链(全盘 VCT)直接走;
- 不输:闸门过滤(对方单线保证、融合点、反四链),只留合法候选;
- 区域剪枝:区域按三源种子划分(强匹配底片 k<=2 / 数量特征热点 / 对方最新
  落子邻域),学来的关系特征分排序,保留与最优分差阈值内的区域(K 不写死),
  合法候选收敛到这几个区域 —— "策略用来剪枝,算力集中推理";
- 集中推理:对剪枝后的候选做受限 VCT(进攻域 = 选中区域,节点预算加大,
  单步思考上限 1 分钟);
- 可能赢:剪枝后的候选上按平坦权重评分 = 点级特征(19)+ 区域关系特征(33),
  取最高者;权重由 Rapfi 蒸馏的启发式搜索得到(experiments/search_weights.py),
  无权重时整体退化为 FrontierAgent。

last_insight 记录本次决策的前五候选与分数,供网页展示决策依据。
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from .board import BLACK, EMPTY, WHITE, Board, opponent
from .features import (POINT_FEATURE_NAMES, extract_features,
                       merged_features)
from .filmsig import fine_key, signature_key
from .frontier_agent import FrontierAgent
from .fusion import four_chain_win
from .match import live_films, nearby_empties, potential_films, win_squares_of
from .relations import directed_domain, film_instances
from .spec_agent import BIG
from .vct import dag_stats, layer_profile, run_length, threat_moves, vct_win

DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[1] / "external" / "rl_weights.json"
)

N_CELL = len(POINT_FEATURE_NAMES)  # 点级特征数(20 原始 + 30 底片联合)

# 开局书:直指开局前 4 手,黑(7,7)→白(7,8)→黑(6,7)→白(6,6)。
# 黑第2手 (6,9) 冠军线由 vizplay.opening_book 表接管(think 层),书保持
# 原样 → 白 RL1.0 对 (6,9) 书失配走模型 (6,8),与 451 冠军线一致。
OPENING = [
    (BLACK, (7, 7)),
    (WHITE, (7, 8)),
    (BLACK, (6, 7)),
    (WHITE, (6, 6)),
]


class RlAgent(FrontierAgent):
    def __init__(self, player: int, expect_layer: int = 4, window: int = 9,
                 weights_path=None, temperature: float = 0.0,
                 region_margin: float = 0.5, region_cap: int = 8,
                 deep_budget: int = 8000, chain_depth: int = 5,
                 chain_nodes: int = 800, deep_leaf_H: int = 5,
                 deep_leaf_nodes: int = 6000, leaf_topk: int = 3):
        super().__init__(player, expect_layer, window)
        self.geo_patches: dict = {}       # 细落子指纹槽族(加性,零=中立)
        self.geo_family = False
        self.weights, self.mean, self.std = self._load(weights_path)
        self.temperature = temperature
        self.region_margin = region_margin  # 区域剪枝:分差阈值(学来的分自适应选几个,K 不写死)
        self.region_cap = region_cap        # 算力保险上限
        self.deep_budget = deep_budget      # 聚焦深推理的 VCT 节点预算(≤1 分钟上限)
        self.chain_depth = chain_depth      # 双方推理预算对称:对方链检查深度(= 我方进攻链深度)
        self.chain_nodes = chain_nodes      # 双方推理预算对称:对方链检查节点(= 我方进攻链节点)
        self.deep_leaf_H = deep_leaf_H      # 叶子比例贡献:沿偏好域的深叶展开层数
        self.deep_leaf_nodes = deep_leaf_nodes  # 深叶节点预算(每候选)
        self.leaf_topk = leaf_topk          # 只对评分最高的前 k 个候选做深叶
        self.leaf_rho = 0.0                 # 深叶胜率加权比例(由权重文件装载,可训练)
        self.gold_w = None                  # 多尺度含金量置信 w(我方/敌方 × s2/3/4)
        self.gold_rho = 0.0                 # 含金量加性权重(最终分 = base + ρ·含金量)
        self.profile_w = None               # 分层 profile 特征权重(默认 None=不启用)
        self.profile_H = 4                  # 对方分层 profile 深度
        self.profile_topk = 5               # 只对评分前 k 候选做分层 profile
        self.shape_gamma = 0.0              # 认形记忆加权(叶子局部形经验)
        self._shape_mem = None              # {形: 经验胜率}(惰性装载)
        self.value_w = None                 # V̂ 预计分数权重(17 维,默认 None)
        self.vhat_scale = 0.0               # V̂ 加性系数(默认 0 = 不启用)
        self.last_insight: list = []      # [(score, [r, c]), ...] 前五候选
        self.last_audit: dict | None = None  # 每步决策审计(在线教学用,行为中立)

    # ---- 权重 ----

    def _load(self, path):
        p = Path(path) if path else DEFAULT_WEIGHTS
        if not p.exists():
            return None, None, None
        data = json.loads(p.read_text(encoding="utf-8"))
        self.patches: dict = {}
        for entry in data.get("patches", []):
            sig = entry["sig"]
            key = int(sig[0]) if isinstance(sig, list) and sig else int(sig)
            self.patches[key] = float(entry.get("w", 0.0))
        self.rel_weights: dict = dict(data.get("rel_weights", {}))
        am = data.get("attention_margin")
        if am is not None:
            self.region_margin = float(am)   # 注意力训练学到的区域分差阈值
        lr = data.get("leaf_rho")
        if lr is not None:
            self.leaf_rho = float(lr)        # 叶子比例贡献的加权比例(可训练)
        gw = data.get("gold_w")
        if gw:
            self.gold_w = [float(x) for x in gw]
        gr = data.get("gold_rho")
        if gr is not None:
            self.gold_rho = float(gr)        # 含金量加性权重
        pw_ = data.get("profile_w")
        if pw_:
            self.profile_w = [float(x) for x in pw_]
        sg = data.get("shape_gamma")
        if sg is not None:
            self.shape_gamma = float(sg)
        vw = data.get("value_w")
        if vw:
            self.value_w = [float(x) for x in vw]
        vs = data.get("vhat_scale")
        if vs is not None:
            self.vhat_scale = float(vs)
        self.geo_patches = {}
        for e in data.get("geo_patches", []):
            sig = e["sig"]
            key = int(sig[0]) if isinstance(sig, list) and sig else int(sig)
            self.geo_patches[key] = float(e.get("w", 0.0))
        self.geo_family = bool(self.geo_patches)
        return data["weights"], data["mean"], data["std"]

    def _shape_prior(self, sig: int) -> float:
        if self._shape_mem is None:
            try:
                p = Path(__file__).resolve().parents[1] / "external" / \
                    "leaf_shapes.json"
                raw = json.loads(p.read_text(encoding="utf-8"))
                self._shape_mem = {int(k): v["wins"] / v["n"]
                                   for k, v in raw.items()}
            except Exception:
                self._shape_mem = {}
        return float(self._shape_mem.get(sig, 0.5))

    def _has_region_weights(self) -> bool:
        return self.weights is not None and len(self.weights) > N_CELL

    def _dot(self, w, mu, sd, feats) -> float:
        s = 0.0
        for wi, mi, si, x in zip(w, mu, sd, feats):
            if si > 1e-9:
                x = (x - mi) / si
            s += wi * x
        return s

    # ---- 闸门遍历(choose 与 distill 共用) ----

    def evaluate(self, board: Board, fast: bool = False):
        """返回 (wins, legal, fallback, pushes, my_films, their_films, weight)。

        - wins     [(关键元组, weight, m)]:一手成五/融合 <=2(直接必胜层);
        - legal    [m]:过不输闸门的合法候选(可能赢层,交给学习评分);
        - fallback [(min_k, weight, m)]:必输拖延层;
        - pushes   {m: _push 元组}:含特征所需的 breadth/fastest/ratio 等。
        fast=True 跳过 VCT 升级(蒸馏数据采集用,省时)。
        """
        me, opp = self.player, opponent(self.player)
        my_films = live_films(board, me, self.expect_layer, window=self.window)
        their_films = live_films(board, opp, self.expect_layer, window=self.window)
        my_win = win_squares_of(my_films)
        if my_win:
            sq = min(my_win)
            return [((0, 0, 0), 0, sq)], [], [], {}, my_films, their_films, {}

        weight: dict = defaultdict(float)
        their_squares: set = set()
        for f in potential_films(board, me, self.expect_layer):
            for sq in f.squares:
                weight[sq] += f.k
        for f in potential_films(board, opp, self.expect_layer):
            for sq in f.squares:
                weight[sq] += 5 - f.k
                their_squares.add(sq)
        candidates = list(weight) if weight else nearby_empties(board)

        wins, legal, fallback, pushes = [], [], [], {}
        memo: dict = {}
        for m in sorted(candidates):
            push = self._push(board, m, their_films, their_squares, memo,
                              use_vct=not fast)
            pushes[m] = push
            (their_k, their_attack, their_chain, their_breadth,
             my_attack, my_four, breadth, fastest, ratio, leaves, my_k) = push

            # 必胜层:有证明的速胜(融合 <=2),且竞速不慢于对方(等号我先到)。
            # T_my = 我方成五总步数最小值(融合倒计时含本手 / 单线 k+1);
            # T_their = 对方总步数最小值(单线 k / 融合 / 强制链)。
            t_my = BIG
            if my_attack is not None:
                t_my = min(t_my, my_attack)
            if my_k != BIG:
                t_my = min(t_my, 1 + my_k)
            t_their = min(their_k, their_attack, their_chain)

            if my_attack is not None and my_attack <= 2 and t_my <= t_their:
                wins.append((-t_my, weight[m], m))
                continue
            # 不输层 = 存在性安全(规格修订,竞速不是安全门槛):
            # 本手落下后,对方在当前算力下无可见强制胜 ——
            #   their_k      >= 3   对方无 2 步内强五(guarantee 已含防守);
            #   their_attack >= 3   对方无融合速胜(双威胁点);
            #   their_chain  >= BIG 对方无可证反四链(300 节点 VCT)。
            # 不安全的候选全部进 fallback,由定向逆向分析的防守通道
            # (反向 VCT)裁决反威胁类防守;竞速快慢只作为"可能赢"层
            # 评分特征的一部分,不再决定生死。
            if their_k >= 3 and their_attack >= 3 and their_chain >= BIG:
                legal.append(m)
            fallback.append((t_their, weight[m], m))
        return wins, legal, fallback, pushes, my_films, their_films, weight

    # ---- 区域剪枝 ----

    def _region_info(self, board, legal):
        """底片聚类 + 区域关系特征。返回 (regions, cell_region, region_feats)。

        无区域或权重不含关系部分时 regions 为 []。"""
        if not self._has_region_weights() or not legal:
            return [], {}, {}
        from .relations import cluster_regions, region_features

        regions = cluster_regions(board)
        if not regions:
            return [], {}, {}
        cell_region: dict = {}
        region_feats: dict = {}
        for idx, reg in enumerate(regions):
            region_feats[idx] = region_features(board, self.player, reg)
            for c in reg.cells:
                cell_region.setdefault(c, idx)
        return regions, cell_region, region_feats

    def _focus(self, board, legal):
        """区域剪枝(底片交互值模型):簇分数 = 簇内最大底片交互值 V。

        V_i = Σ_j W[rel(i,j), k_i, k_j](双方,有向贡献,未训练槽位用先验);
        值最大的底片所关联的簇最值得选。K 不写死:保留与最优分差
        region_margin 内且非负的簇;无入选回退全部 legal。
        """
        regions, cell_region, region_feats = self._region_info(board, legal)
        if not regions:
            return legal, {}
        from .relations import film_values

        values = film_values(board, self.player, self.rel_weights)
        v_of = {id(f): v for v, f in values}
        scored = []
        for idx, reg in enumerate(regions):
            s = max((v_of.get(id(f), 0.0) for f in reg.films), default=0.0)
            scored.append((s, idx))
        scored.sort(reverse=True)
        best = scored[0][0]
        top = []
        for s, idx in scored:
            if s >= best - self.region_margin and s >= 0:
                top.append(idx)
            if len(top) >= self.region_cap:
                break
        if not top:
            return legal, {}                       # 无正值簇:不剪枝
        focused = [m for m in legal if cell_region.get(m) in set(top)]
        if not focused:
            return legal, {}
        feats = {m: region_feats[cell_region[m]] for m in focused}
        return focused, feats

    # ---- 走子 ----

    def _opening(self, board: Board):
        """开局书:前缀吻合时按书落子,否则 None(交回模型)。"""
        ply = len(board.moves)
        if ply >= len(OPENING):
            return None
        for i in range(ply):
            if (board.moves[i][0], board.moves[i][1]) != OPENING[i][1]:
                return None
        player, cell = OPENING[ply]
        if player != self.player or board.get(cell[0], cell[1]) != EMPTY:
            return None
        return cell

    def choose(self, board: Board):
        if self.weights is None:
            return super().choose(board)
        self.last_audit = {"stage": None, "mv": None, "ply": len(board.moves),
                           "player": self.player}
        move = self._opening(board)
        if move is not None:
            self.last_audit["stage"] = "book"
            self.last_audit["mv"] = list(move)
            return move
        me, opp = self.player, opponent(self.player)
        their_films0 = live_films(board, opp, self.expect_layer, window=self.window)
        if not any(f.k == 1 for f in their_films0):
            chain = four_chain_win(board, me, 4)
            if chain is None:
                chain = vct_win(board, me, 5, budget={"n": 800})
            if chain is not None:
                self.last_audit["stage"] = "chain"
                self.last_audit["mv"] = list(chain[1])
                return chain[1]

        wins, legal, fallback, pushes, my_films, their_films, weight = (
            self.evaluate(board)
        )
        if wins:
            mv = max(wins)[2]
            self.last_audit["stage"] = "wins"
            self.last_audit["mv"] = list(mv)
            return mv
        top_films, dom = directed_domain(board, me, self.rel_weights)
        if legal:
            # 定向逆向分析·进攻(超剪枝):只对最重要的前几个底片域做深推
            if dom:
                hit = vct_win(board, me, 5, budget={"n": self.deep_budget},
                              restrict=dom)
                if hit is not None:
                    self.last_audit["stage"] = "dom_vct"
                    self.last_audit["mv"] = list(hit[1])
                    return hit[1]
            focused, region_feats = self._focus(board, legal)
            # 集中算力:剪枝生效且聚焦域内存在威胁前景(造四/攻击值)时,
            # 才做受限 VCT 深推 —— 无前景直接跳过,避免烧穿预算
            if len(focused) < len(legal):
                promising = any(
                    (v is not None and v <= 5)
                    or run_length(board, cell[0], cell[1], me) >= 4
                    for v, cell in threat_moves(board, me, restrict=set(focused))
                )
                if promising:
                    hit = vct_win(board, me, 5, budget={"n": self.deep_budget},
                                  restrict=set(focused))
                    if hit is not None:
                        self.last_audit["stage"] = "focus_vct"
                        self.last_audit["mv"] = list(hit[1])
                        return hit[1]
            zeros = [0.0] * (len(self.weights) - N_CELL)
            insts = film_instances(board)
            cov_gold = self._gold_cov(board, me) if (self.gold_w
                                                     and self.gold_rho) else None
            scored = []
            for m in focused:
                cell = extract_features(board, me, m, my_films, their_films,
                                        pushes[m], insts)
                feats = merged_features(cell, region_feats.get(m, zeros))
                s = self._dot(self.weights, self.mean, self.std, feats)
                s += self.patches.get(signature_key(board, me, m, insts), 0.0)
                if self.geo_family:
                    s += self.geo_patches.get(fine_key(board, me, m, insts),
                                              0.0)
                if cov_gold is not None:      # 多尺度含金量加性评分(训练于冠军格)
                    s += self.gold_rho * self._gold_value(m, cov_gold, me)
                scored.append((s, (-m[0], -m[1]), m))
            scored.sort(reverse=True)
            self.last_insight = [(round(s, 3), [m[0], m[1]]) for s, _, m in scored[:5]]
            if self.temperature > 0:
                vals = [s for s, _, _ in scored]
                m = max(vals)
                ex = [math.exp((s - m) / self.temperature) for s in vals]
                total = sum(ex)
                r = random.random() * total
                for (s, _, m), e in zip(scored, ex):
                    r -= e
                    if r <= 0:
                        self.last_audit["stage"] = "temp"
                        self.last_audit["mv"] = list(m)
                        return m
            if self.leaf_rho and dom:
                self._leaf_adjust(scored, board, me, dom)
            if self.profile_w and dom:
                for idx in range(min(self.profile_topk, len(scored))):
                    s, _tie, m = scored[idx]
                    feats = self._profile_feats(board, me, m, dom)
                    add = sum(wv * fv for wv, fv in zip(self.profile_w,
                                                        feats))
                    scored[idx] = (s + add, _tie, m)
                scored.sort(key=lambda t: t[0], reverse=True)
            mv = scored[0][2]             # 区内最优:评分器(+深叶比例贡献)
            self.last_audit["stage"] = "score"
            self.last_audit["mv"] = list(mv)
            self.last_audit["legal"] = [list(x) for x in legal]
            self.last_audit["focused"] = [list(x) for x in focused]
            self.last_audit["focused_eq_legal"] = len(focused) == len(legal)
            # 评分快照(线上调参用:分数含当时 patches/gold,行为中立)
            self.last_audit["scored"] = [
                (round(float(s), 6), [m[0], m[1]]) for s, _, m in scored]
            return mv
        self.last_insight = []
        mv = self._directed_defense(board, fallback, dom)
        if mv is not None:
            self.last_audit["stage"] = "defense"
            self.last_audit["mv"] = list(mv)
            return mv
        mv = max(fallback)[2]
        self.last_audit["stage"] = "fallback"
        self.last_audit["mv"] = list(mv)
        return mv

    def _push(self, board: Board, m, their_films, their_squares, memo,
              use_vct: bool = True):
        """覆盖:对方反四链检查用对称预算(chain_depth/chain_nodes,默认 5/800
        = 我方进攻链的深度与预算 —— 用户规格"双方预算一样")。"""
        return super()._push(board, m, their_films, their_squares, memo,
                             use_vct=use_vct, chain_depth=self.chain_depth,
                             chain_nodes=self.chain_nodes)

    def _leaf_adjust(self, scored, board, me, dom):
        """叶子比例贡献(用户规格 B,与模拟同性质的推理叶子,更远更特殊):
        对评分最高的前 leaf_topk 个候选,沿正向偏好域 dom(环境更新后的
        决策:top 底片 = 未来要发展的路径)展开深叶,深叶胜率以 leaf_rho
        加权进分 —— 最终分 = 评分 + ρ·(深叶胜率 − 0.5)。ρ 可训练。
        shape_gamma 非零时叠加认形:深叶统计改用 dag_shape_stats,
        再加 γ·(叶形经验均值 − 0.5)(叶子局部形之前出现过 → 单独加分)。
        """
        from .vct import dag_shape_stats
        for idx in range(min(self.leaf_topk, len(scored))):
            s, _tie, m = scored[idx]
            add = 0.0
            if self.vhat_scale:
                board.place(m[0], m[1], me)
                try:
                    add += self.vhat_scale * (self._vhat(board, me) - 0.5)
                finally:
                    board.undo()
            if self.shape_gamma:
                res = dag_shape_stats(
                    board, me, m, self.deep_leaf_H, self.deep_leaf_nodes,
                    fast=True, restrict=dom,
                    prior_fn=self._shape_prior, budget=200)
                if res is not None:
                    add += (self.leaf_rho * (res[0] - 0.5)
                            + self.shape_gamma * (res[3]["mean"] - 0.5))
                    scored[idx] = (s + add, _tie, m)
                    continue
            res = dag_stats(board, me, m, self.deep_leaf_H,
                            self.deep_leaf_nodes, fast=True, restrict=dom)
            if res is None:
                continue
            add += self.leaf_rho * (res[0] - 0.5)
            scored[idx] = (s + add, _tie, m)
        scored.sort(key=lambda t: t[0], reverse=True)

    def _gold_cov(self, board, me):
        """多尺度覆盖:每方每层(s=2/3/4)的无敌 5 窗格集(直接匹配,不判非法)。"""
        opp = opponent(me)
        cov = {me: {s: [] for s in (2, 3, 4)},
               opp: {s: [] for s in (2, 3, 4)}}
        for idx, line in enumerate(board.lines()):
            vals = [board.get(r, c) for (r, c) in line]
            for i in range(len(line) - 4):
                span = vals[i:i + 5]
                n_me = span.count(me)
                n_opp = span.count(opp)
                cells = set(line[i:i + 5])
                if n_opp == 0 and n_me in (2, 3, 4):
                    cov[me][n_me].append(cells)
                if n_me == 0 and n_opp in (2, 3, 4):
                    cov[opp][n_opp].append(cells)
        return cov

    def _gold_value(self, m, cov, me):
        """含金量 = Σ w[方][s]·覆盖窗数(置信参数从权重装载)。"""
        counts = []
        for pl in (me, opponent(me)):
            for s in (2, 3, 4):
                counts.append(sum(1 for ws in cov[pl][s] if m in ws))
        return float(sum(w * c for w, c in zip(self.gold_w, counts)))

    def _profile_feats(self, board, me, m, dom):
        """分层 profile 特征(与训练同口径):
        [对方胜叶·消耗1手/2手/≥3手 计数(归一), 我方深叶胜率(restrict=dom)]。"""
        opp = opponent(me)
        board.place(m[0], m[1], me)
        try:
            rp = layer_profile(board, opp, H=self.profile_H, node_cap=1500,
                               fast=False, restrict=dom)
            own = dag_stats(board, me, m, 5, 6000, fast=True, restrict=dom)
        finally:
            board.undo()
        feats = [0.0, 0.0, 0.0]
        if rp is not None:
            total, _wins, layers = rp
            den = 1.0 + total
            for d_rem, (wd, _ld) in layers.items():
                cons = self.profile_H - d_rem      # 已消耗的对方手数
                if cons <= 1:
                    feats[0] += wd / den
                elif cons == 2:
                    feats[1] += wd / den
                else:
                    feats[2] += wd / den
        feats.append(own[0] if own is not None else 0.5)
        return feats

    def _vhat(self, board, me) -> float:
        """预计分数 V̂(17 维,与 train_value 同口径);未启用返回 0.5。"""
        if self.value_w is None:
            return 0.5
        f = [0.0] * 17
        for pl, off in ((me, 0), (opponent(me), 8)):
            for film in live_films(board, pl, 4):
                if film.k <= 4:
                    f[off + film.k - 1] += 1.0
            for p in potential_films(board, pl, 4):
                if p.k <= 4:
                    f[off + 4 + p.k - 1] += 1.0
        mine = sum(1 for row in board.grid for c in row if c == me)
        theirs = len(board.moves) - mine
        f[16] = mine - theirs
        return float(sum(wv * x for wv, x in zip(self.value_w, f)))

    def _directed_defense(self, board, fallback, dom):
        """定向逆向分析·防守:域内候选逐手落子后用反向 VCT 验证。

        对方在最重要底片域内的强制胜(vct_win(opp, 5))若被某候选驳掉
        (落子后对手无胜),该候选进入"真防守"池,按 (t_their, weight) 取
        最大;全部驳不掉或域为空时返回 None(交回原 fallback 拖延)。
        """
        if not dom:
            return None
        opp = opponent(self.player)
        best = None
        checked = 0
        for (t, w, m) in fallback:
            if m not in dom:
                continue
            board.place(m[0], m[1], self.player)
            try:
                hit = vct_win(board, opp, 5, budget={"n": 900}, restrict=dom)
            finally:
                board.undo()
            if hit is None:
                if best is None or (t, w) > best[0]:
                    best = ((t, w), m)
            checked += 1
            if checked >= 8:
                break
        return best[1] if best is not None else None
