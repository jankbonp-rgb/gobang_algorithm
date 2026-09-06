"""末端前推(retrograde / backward induction)与底片(film)。

对每个一维窗口局势 s 做局部 AND/OR 求解:
    guarantee(s, US)   轮我方走:本窗口内保证成五所需的我方最少手数;无保证为 None。
    guarantee(s, THEM) 轮对方走:对方可下窗口内任意空点,也可"弃权"(等价于下在窗外),
                       仍然保证时返回手数,否则 None。

语义要点:
- 我方选点是 OR 分支,对方应对是 AND 分支 —— 单条轨迹不构成证明,
  底片的"倒计时保证"必须对对方的每个局部防守都成立,这正是这里算出的东西。
- 对方下在窗外对本窗口等价于弃权,所以局部保证对"对方在本窗口内的防守"是完备的;
  它故意不覆盖对方在别处的反威胁抢先手 —— 那部分交给全局竞速判断(见 agent)。
- 库按层组织:k = 倒计时(还差我方几手),就是讨论里 DAG 的"层"。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

US, THEM = 0, 1


def has_five(s: str) -> bool:
    return "MMMMM" in s


@lru_cache(maxsize=None)
def guarantee(s: str, mover: int):
    if has_five(s):
        return 0
    if mover == US:
        best = None
        for i, ch in enumerate(s):
            if ch != "E":
                continue
            k = guarantee(s[:i] + "M" + s[i + 1 :], THEM)
            if k is not None and (best is None or k + 1 < best):
                best = k + 1
        return best
    # 对方回合:弃权(下在窗外)也是一个选项
    worst = guarantee(s, US)
    if worst is None:
        return None
    for i, ch in enumerate(s):
        if ch != "E":
            continue
        k = guarantee(s[:i] + "B" + s[i + 1 :], US)
        if k is None:
            return None
        if k > worst:
            worst = k
    return worst


def best_first_moves(s: str) -> list[int]:
    """倒计时的第一步可选点(轨迹的起点集合,OR 分支)。"""
    k = guarantee(s, US)
    if not k:
        return []
    out = []
    for i, ch in enumerate(s):
        if ch != "E":
            continue
        if guarantee(s[:i] + "M" + s[i + 1 :], THEM) == k - 1:
            out.append(i)
    return out


@lru_cache(maxsize=None)
def leaf_stats(s: str, mover: int) -> tuple[int, int]:
    """全展开子树(双方在窗口内任意落子,不含弃权)的 (叶子数, 我方成五叶子数)。

    对应讨论中的"同族叶子里赢的比例 / 节点数"裁决。注意:叶子在对抗下
    并非等概率,这个统计天然偏乐观,只应当作最末位的平手裁决用。
    """
    if has_five(s):
        return (1, 1)
    empties = [i for i, ch in enumerate(s) if ch == "E"]
    if not empties:
        return (1, 0)
    total = wins = 0
    stone = "M" if mover == US else "B"
    nxt = THEM if mover == US else US
    for i in empties:
        l, w = leaf_stats(s[:i] + stone + s[i + 1 :], nxt)
        total += l
        wins += w
    return (total, wins)


@dataclass(frozen=True)
class Film:
    """底片:匹配到的局部局势 + 它的倒计时信息。

    pattern     窗口局势(我方视角字符串)
    k           倒计时:还需我方几手(DAG 的层号)
    them_proof  对方先手也挡不住时为 True(如活四);False 表示需要我方先手(如冲四、活三)
    squares     轨迹第一步的棋盘坐标(OR 分支)
    line_index  所在线号(用于落子后的增量重匹配)
    coords      与 pattern 逐位对齐的棋盘坐标(补位处为 None),用于 VCT 的杀死格映射
    """

    pattern: str
    k: int
    them_proof: bool
    squares: tuple
    line_index: int = -1
    coords: tuple | None = None
