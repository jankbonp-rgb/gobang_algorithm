"""忠实版走子:按讨论稿逐条字面实现,不做规格之外的寻优。

对应关系:
1. 必胜 > 不输 > 可能赢 —— 有一手成五先走(必胜;这也是规格自身的推论,
   所以点优先级"对方 > 我方"不适用于这一步);
2. 不输 = 预期层之内不进入对方任何底片:落子后对方仍有倒计时 k<=2 的
   保证(冲四/活四在手,或活三将转活四)则该点非法。唯一例外:该子让我方
   同时出现两个成五点(对方一手挡不住,按手数我方先到)——"必胜>不输"的推论;
3. 候选点 = 双方底片轨迹上的点,优先级:交集 > 对方 > 我方;
4. 可能赢 = 在合法点中,取这手棋指向的同族(经过该点的居中窗口)里
   赢的叶子比例最多者,比例相同取叶子(节点)数多者 —— 原文的字面直译;
5. 规格未定义的两处用最小回退:无任何底片可匹配时(如开局),候选退为
   已有棋子邻域,仍用第 4 条裁决;所有点都非法(必输)时选让对方倒计时
   最慢的点。除此之外没有中心偏好、没有竞速打分。
"""
from __future__ import annotations

from .board import Board, opponent
from .language import WINDOW, board_windows
from .match import live_films, nearby_empties, win_squares_of
from .retrograde import THEM, leaf_stats

BIG = 99
_CENTER = WINDOW // 2


class SpecAgent:
    def __init__(self, player: int, expect_layer: int = 4, window: int = 9):
        self.player = player
        self.expect_layer = expect_layer
        self.window = window  # 算力档位:末端前推的窗口长度(9/11/13)

    def choose(self, board: Board) -> tuple[int, int]:
        me, opp = self.player, opponent(self.player)
        my_films = live_films(board, me, self.expect_layer, window=self.window)
        their_films = live_films(board, opp, self.expect_layer, window=self.window)

        my_win = win_squares_of(my_films)
        if my_win:
            return min(my_win)

        their_sq = {sq for f in their_films for sq in f.squares}
        my_sq = {sq for f in my_films for sq in f.squares}
        ranked = (
            [(2.0, m) for m in their_sq & my_sq]
            + [(1.0, m) for m in their_sq - my_sq]
            + [(0.5, m) for m in my_sq - their_sq]
        )
        if not ranked:
            ranked = [(0.0, m) for m in nearby_empties(board)]

        legal, fallback = [], []
        for rank, m in sorted(ranked, key=lambda x: x[1]):
            their_k, my_wins, ratio, leaves = self._after(board, m, their_films)
            entry = (rank, ratio, leaves, m)
            if their_k >= 3 or (their_k == 2 and my_wins >= 2):
                legal.append(entry)
            fallback.append((their_k, rank, ratio, leaves, m))

        if legal:
            rank, ratio, leaves, m = max(
                legal, key=lambda e: (e[0], e[1], e[2], (-e[3][0], -e[3][1]))
            )
            return m
        # 必输局面:规格未定义,选让对方倒计时最慢的点尽量拖延
        return max(fallback, key=lambda e: (e[0], e[1], e[2], e[3]))[4]

    def _after(self, board: Board, m, their_films):
        """落子 m 之后:对方最快倒计时 / 我方成五点数 / 这手棋指向的同族叶子统计。"""
        me, opp = self.player, opponent(self.player)
        touched = set(board.lines_through(*m))
        board.place(m[0], m[1], me)
        try:
            their_after = [f for f in their_films if f.line_index not in touched]
            their_after += live_films(
                board, opp, self.expect_layer, touched, window=self.window
            )
            their_k = min((f.k for f in their_after), default=BIG)

            my_touched = live_films(
                board, me, self.expect_layer, touched, window=self.window
            )
            my_wins = len(win_squares_of(my_touched))

            # 叶子裁决固定用窗口 9(平手裁决不需要大窗口,避免子树统计爆炸)
            ratio, leaves = 0.0, 0
            for _idx, pattern, coords in board_windows(board, me, touched, window=9):
                if coords[_CENTER] != m:
                    continue
                total, wins = leaf_stats(pattern, THEM)
                if (wins / total, total) > (ratio, leaves):
                    ratio, leaves = wins / total, total
            return their_k, my_wins, ratio, leaves
        finally:
            board.undo()
