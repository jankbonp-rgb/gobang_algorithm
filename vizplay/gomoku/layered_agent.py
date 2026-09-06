"""预测层版(规格 v2 + 交点组合层):所有层的底片加权 + 混合结构的攻防。

规格来源(用户原话的形式化):
- "选择下一个当前合法底图及其步数的一个符合的函数最值的位置"
  → 在合法(不输)候选中取 score(m) 最大者;
- "自己的风格保持越是后面步数的底图的权重越大"
  → 我方潜在底片按 α(k) = k 加权(k = 剩余步数,越深越重);
- "别人的则是越是近期的越要防守"
  → 对方潜在底片按 β(k) = 5 - k 加权(越临近越重);
- score(m) = Σα + Σβ,m 同时落在双方轨迹上时两边都计入,
  v1 的"交点 > 对方 > 我方"被这个函数自然涵盖;
- 潜在底片不要求局部保证("出现底片且轨迹为空即进入倒计时"的直译),
  于是"提前赢的那些层"(k=1,2)与深层(k=3,4)一起参与计算;
- 交点组合层(fusion,混合结构 = 两条线在交点的合法融合):
  进攻上,我方融合倒计时 <=2(双四/活四级)视同必胜,
  =3(四三/双三)且不慢于对方时优先;防守上,落子后不许留给对方
  <=2 的融合点 —— 这是"不输"从单线保证到混合结构的扩展;
- 其余层级不变:一手成五先走;落子后不许留给对方 k<=2 的单线保证;
  同分时沿用 v1 的同族叶子胜率裁决。
预期层(expect_layer)限定参与加权的最大层号。
"""
from __future__ import annotations

from collections import defaultdict

from .board import EMPTY, Board, opponent
from .fusion import attack_countdown, attack_value_at
from .language import board_windows
from .match import live_films, nearby_empties, potential_films, win_squares_of
from .retrograde import THEM, leaf_stats
from .spec_agent import BIG, SpecAgent


class LayeredAgent(SpecAgent):
    def choose(self, board: Board):
        me, opp = self.player, opponent(self.player)
        my_films = live_films(board, me, self.expect_layer, window=self.window)
        their_films = live_films(board, opp, self.expect_layer, window=self.window)
        my_win = win_squares_of(my_films)
        if my_win:
            return min(my_win)

        weight: dict = defaultdict(float)
        their_squares: set = set()
        for f in potential_films(board, me, self.expect_layer):
            for sq in f.squares:
                weight[sq] += f.k  # α(k)=k:自己越深的底片越重
        for f in potential_films(board, opp, self.expect_layer):
            for sq in f.squares:
                weight[sq] += 5 - f.k  # β(k)=5-k:对方越临近越要防
                their_squares.add(sq)
        candidates = list(weight) if weight else nearby_empties(board)

        wins, legal, fallback = [], [], []
        for m in sorted(candidates):
            their_k, their_attack, my_attack, ratio, leaves = self._after_fused(
                board, m, their_films, their_squares
            )
            # 双四/活四(我方全局第3手成五)只在对方没有一手成五(their_k>=2,
            # 对方最快也要全局第4手)时才是真必胜 —— 竞速按手数比较
            if my_attack is not None and my_attack <= 2 and their_k >= 2:
                wins.append((-my_attack, weight[m], m))
                continue
            if their_k >= 3 and their_attack >= 3:
                # 四三/双三:平速时我方先到(我方融合已含本手,对方还没走)
                win_soon = 1 if (my_attack is not None and my_attack <= their_attack) else 0
                legal.append((win_soon, weight[m], ratio, leaves, (-m[0], -m[1]), m))
            fallback.append((min(their_k, their_attack), weight[m], m))

        if wins:
            return max(wins)[2]
        if legal:
            return max(legal)[5]
        # 必输局面:规格未定义,选让对方(单线或融合)倒计时最慢的点拖延
        return max(fallback)[2]

    def _after_fused(self, board: Board, m, their_films, their_squares):
        """落子 m 之后:(对方单线最快 k, 对方最快融合点, 我方融合值, 叶子统计)。"""
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

            their_attack = BIG
            for cell in sorted(their_squares):
                if board.get(cell[0], cell[1]) != EMPTY:
                    continue
                v = attack_countdown(board, opp, cell)
                if v is not None and v < their_attack:
                    their_attack = v
                    if v <= 2:
                        break

            ratio, leaves = 0.0, 0
            for _idx, pattern, coords in board_windows(board, me, touched, window=9):
                if coords[4] != m:
                    continue
                total, w = leaf_stats(pattern, THEM)
                if (w / total, total) > (ratio, leaves):
                    ratio, leaves = w / total, total
            return their_k, their_attack, my_attack, ratio, leaves
        finally:
            board.undo()
