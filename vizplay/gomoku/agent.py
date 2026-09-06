"""增强版走子(对照组):在讨论稿规则之上叠加了三处规格之外的寻优。

与忠实版(spec_agent.SpecAgent)的差异,即被用户指出"优化掉本质"的部分:
1. 竞速分级评分:their_k / my_k 作为打分项参与排序(规格里"不输"只是过滤);
2. 候选点为全邻域逐点重打分 + 中心偏好(规格里候选来自底片轨迹);
3. 叶子胜率规则被降为末位平手裁决(规格里它是"可能赢"阶段的主选择器)。

保留此实现仅用于 A/B 对照,量化这三处偏离带来的棋力差。
"""
from __future__ import annotations

from .board import Board, opponent
from .match import live_films, nearby_empties, win_squares_of
from .retrograde import THEM, Film, leaf_stats

BIG = 99


class Agent:
    def __init__(self, player: int, expect_layer: int = 4, window: int = 9):
        self.player = player
        self.expect_layer = expect_layer  # 预期层:只在这一层以内的底片上集中算力
        self.window = window

    def choose(self, board: Board) -> tuple[int, int]:
        me, opp = self.player, opponent(self.player)
        my_films = live_films(board, me, self.expect_layer, window=self.window)
        their_films = live_films(board, opp, self.expect_layer, window=self.window)

        my_win = win_squares_of(my_films)
        if my_win:
            return min(my_win)

        their_win = win_squares_of(their_films)
        if their_win:
            # 必须堵(v1 不做反四抢手数);多个堵点时选给自己带来最多后续的那个
            pool = sorted(their_win)
        else:
            pool = nearby_empties(board)

        their_squares = {sq for f in their_films for sq in f.squares}
        my_squares = {sq for f in my_films for sq in f.squares}

        best_move, best_score = None, None
        for m in pool:
            if m in their_squares and m in my_squares:
                priority = 2.0
            elif m in their_squares:
                priority = 1.0
            elif m in my_squares:
                priority = 0.5
            else:
                priority = 0.0
            score = self._score(board, m, my_films, their_films, priority)
            if best_score is None or score > best_score:
                best_move, best_score = m, score
        return best_move

    def _score(
        self,
        board: Board,
        m: tuple[int, int],
        my_films: list[Film],
        their_films: list[Film],
        priority: float,
    ):
        me, opp = self.player, opponent(self.player)
        touched = set(board.lines_through(*m))
        board.place(m[0], m[1], me)
        try:
            # 增量重匹配:只有经过 m 的线上的底片会变
            my_after = [f for f in my_films if f.line_index not in touched]
            my_after += live_films(
                board, me, self.expect_layer, touched, window=self.window
            )
            their_after = [f for f in their_films if f.line_index not in touched]
            their_after += live_films(
                board, opp, self.expect_layer, touched, window=self.window
            )

            fork = len(win_squares_of(my_after)) >= 2  # 对方一手挡不住:活四或双冲四
            my_k = min((f.k for f in my_after if f.them_proof), default=BIG)
            their_k = min((f.k for f in their_after), default=BIG)

            if fork:
                win_class = 2
            elif my_k < their_k:
                win_class = 1
            else:
                win_class = 0
            # 不输:不能留给对方"我方怎么防局部都输"的倒计时(如活三将转活四)
            survive = 1 if (their_k > 2 or win_class == 2) else 0

            ratio, leaves = 0.0, 0
            if my_after:
                best = min(my_after, key=lambda f: (f.k, not f.them_proof))
                if len(best.pattern) == 9:  # 大窗口时跳过叶子统计(避免子树爆炸)
                    total, wins = leaf_stats(best.pattern, THEM)
                    ratio, leaves = wins / total, total

            center = board.size // 2
            centrality = -(abs(m[0] - center) + abs(m[1] - center))
            return (
                survive,
                win_class,
                priority,
                min(their_k, 6),
                -min(my_k, 6),
                ratio,
                leaves,
                centrality,
            )
        finally:
            board.undo()
