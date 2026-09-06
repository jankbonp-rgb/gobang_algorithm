"""首端推理版(规格 v3):从当前局势的活底片出发前推 + 三特征复合评估。

规格来源(用户原话的形式化):
- 末端底片总量太大(半形 3,868,融合条目 7,482,646),无法全体继续前推;
  当前局势本身就是最强的匹配/剪枝 —— 轨迹与位置约束下合法的当前底片很少,
  把算力集中在这些底片(敌我双方)上前推,寻找步数更少的局面;
- 三特征复合评估:
    特征3 不能输或者必胜 —— 权重 100%:硬性闸门。必胜列表(融合<=2 且竞速
        合法)直接走;不输过滤(单线保证与融合点均不许留给对方 <=2)必须过。
    特征1 下一层 DAG 分支上赢的机会多(对方犯错空间大)—— 权重 0.6:
        v3.1 起为净分支:我方可发起进攻倒计时(<=3)的分支点数 减去
        对方的同类分支点数 —— 对手的首端用同一套融合表对称模拟,
        否则对手只以合法/非法的二值形式进入决策,防守没有梯度。
    特征2 越早赢权重越高 —— 权重 0.4:从现在起最快成五的手数,线性归一化。
  复合分 = 0.6 * 净分支 + 0.4 * 早胜度;同分回退 v2 的 α/β 权重与叶子裁决。
"""
from __future__ import annotations

from collections import defaultdict

from .board import EMPTY, Board, opponent
from .fusion import (
    attack_countdown,
    attack_value_at,
    centered_patterns,
    four_chain_win,
)
from .language import board_windows
from .match import live_films, nearby_empties, potential_films, win_squares_of
from .retrograde import THEM, US, guarantee, leaf_stats
from .spec_agent import BIG, SpecAgent
from .vct import vct_win


class FrontierAgent(SpecAgent):
    def choose(self, board: Board):
        me, opp = self.player, opponent(self.player)
        my_films = live_films(board, me, self.expect_layer, window=self.window)
        their_films = live_films(board, opp, self.expect_layer, window=self.window)
        my_win = win_squares_of(my_films)
        if my_win:
            return min(my_win)
        memo: dict = {}
        attack_budget = {"n": 800}
        # 反四链:我方的连续冲四强制胜(对方无一手成五在手时才可发动)。
        # 旧链搜索(快速、保守)先行;未命中时升级完整 VCT(反四嵌套、
        # 第三处链、多线融合),节点预算用尽即放弃(等同旧行为)。
        if not any(f.k == 1 for f in their_films):
            chain = four_chain_win(board, me, 4)
            if chain is None:
                chain = vct_win(board, me, 5, memo=memo, budget=attack_budget)
            if chain is not None:
                return chain[1]

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

        wins, legal, fallback = [], [], []
        for m in sorted(candidates):
            (
                their_k, their_attack, their_chain, their_breadth,
                my_attack, my_four, breadth, fastest, ratio, leaves, my_k,
            ) = self._push(board, m, their_films, their_squares, memo)
            if my_attack is not None and my_attack <= 2 and their_k >= 2:
                wins.append((-my_attack, weight[m], m))
                continue
            # 特征3:不输闸门(权重100%)。their_chain 是对方反四链的静止检查:
            # 对方一旦有强制链,我方倒计时被逐手冻结(双成五点在手除外,
            # 那种情况已从 wins 分支走掉)。对方融合点=3(四三级,局部防不住)
            # 时,合法的竞速资源有两种:我方 <=3 的融合(平速我先到),
            # 或我方这手本身成四 —— 四强制对方应手,对方腾不出手下融合点
            # (偷手数的防守用法),下一回合重新评估。
            races_ok = (
                their_attack > 3
                or (my_attack is not None and my_attack <= 3)
                or my_four
            )
            if their_k >= 3 and their_attack >= 3 and their_chain >= BIG and races_ok:
                win_soon = 1 if (my_attack is not None and my_attack <= their_attack) else 0
                # 特征1:净分支 = 我方下一层赢分支 - 对方的(对手首端的对称模拟)
                b = (min(breadth, 5) - min(their_breadth, 5)) / 5.0
                e = max(0.0, (6.0 - min(fastest, 6)) / 5.0)  # 特征2:早胜度
                composite = 0.6 * b + 0.4 * e
                legal.append(
                    (win_soon, composite, weight[m], ratio, leaves, (-m[0], -m[1]), m)
                )
            fallback.append((min(their_k, their_attack, their_chain), weight[m], m))

        if wins:
            return max(wins)[2]
        if legal:
            return max(legal)[6]
        return max(fallback)[2]

    def _push(self, board: Board, m, their_films, their_squares, memo,
              use_vct: bool = True, chain_depth: int = 4,
              chain_nodes: int = 300):
        """首端前推:落子 m 后,在双方活底片的轨迹上集中评估。

        返回 (对方单线最快k, 对方最快融合点, 对方反四链手数, 对方分支数,
              我方融合值, 我方是否成四, 我方分支数 breadth, 最快成五手数 fastest,
              叶子统计 ratio/leaves, 我方单线最快k)。
        use_vct=False 时跳过 VCT 升级(蒸馏数据采集的轻量闸门)。
        chain_depth/chain_nodes:对方反四链检查的深度与节点预算(默认 4/300,
        RL 引擎覆盖为与我方进攻链对称的 5/800 —— 双方预算一样)。
        """
        me, opp = self.player, opponent(self.player)
        touched = set(board.lines_through(*m))
        board.place(m[0], m[1], me)
        try:
            their_after = [f for f in their_films if f.line_index not in touched]
            their_after += live_films(
                board, opp, self.expect_layer, touched, window=self.window
            )
            their_k = min((f.k for f in their_after), default=BIG)

            my_attack = attack_value_at(board, me, m)
            # 这手是否成四(唯一可能的四必经过 m,因为此前我方无一手成五点)
            my_four = any(
                guarantee(p, US) == 1 for p in centered_patterns(board, me, m)
            )

            their_attack, their_breadth = BIG, 0
            for cell in sorted(their_squares):
                if board.get(cell[0], cell[1]) != EMPTY:
                    continue
                v = attack_countdown(board, opp, cell)
                if v is None:
                    continue
                if v <= 3:
                    their_breadth += 1
                if v < their_attack:
                    their_attack = v
                    if v <= 2:
                        break  # 反正会被闸门拒绝,复合分不再需要

            # 特征1/2:只在我方落子后的活底片轨迹上前推(当前局势=剪枝)
            breadth = 0
            fastest = my_attack if my_attack is not None else BIG
            follow_cells = {
                sq
                for f in potential_films(board, me, self.expect_layer)
                for sq in f.squares
            }
            for cell in follow_cells:
                v = attack_countdown(board, me, cell)
                if v is not None and v <= 3:
                    breadth += 1
                    fastest = min(fastest, v + 1)  # 该分支还需先走 cell 这一手

            # 对方反四链的静止检查(仅对闸门候选计算,懒求值):
            # 旧链搜索先行,未命中时升级完整 VCT(含反四嵌套/第三处链);
            # 逐候选独立节点预算,避免共享预算耗尽后语义漂移(漏检)
            their_chain = BIG
            if their_k >= 3 and their_attack >= 3:
                hit = four_chain_win(board, opp, 4)
                if hit is None and use_vct:
                    hit = vct_win(board, opp, chain_depth, memo=memo,
                                  budget={"n": chain_nodes})
                if hit is not None:
                    their_chain = hit[0]

            ratio, leaves = 0.0, 0
            for _idx, pattern, coords in board_windows(board, me, touched, window=9):
                if coords[4] != m:
                    continue
                total, w = leaf_stats(pattern, THEM)
                if (w / total, total) > (ratio, leaves):
                    ratio, leaves = w / total, total
            # 我方落子后的单线最快 k(统一竞速判据用)
            my_after = live_films(
                board, me, self.expect_layer, touched, window=self.window
            )
            my_k = min((f.k for f in my_after), default=BIG)
            return (
                their_k, their_attack, their_chain, their_breadth,
                my_attack, my_four, breadth, fastest, ratio, leaves, my_k,
            )
        finally:
            board.undo()
