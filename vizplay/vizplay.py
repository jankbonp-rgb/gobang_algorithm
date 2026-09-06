"""思考可视化对弈(独立副本 vizplay/,与训练代码隔离,可整目录回溯)。

功能:
- 与用户对弈 / 模型互博(观战,手动一步步);
- 每步 RL 思考后,棋盘上画出"思考范围":被考虑最多的底片轮廓
  (活底片=实心色块,潜在底片=空心框),颜色深浅 = 该底片被最终采纳的
  概率(由最终候选动作 softmax 映射回其触及的底片);非 RL 引擎只显示
  filmV 相对强度;
- 右侧面板:各底片 V/采纳率 + 前五候选格概率 + 引擎/槽数。

布局:窗口可自由拉伸,棋盘按窗口大小等比缩放居中;点击换算与绘制共用
同一套坐标变换(逻辑棋盘 0..504 + 边距,屏幕任意缩放),杜绝偏移。

兼容性:只读训练产出 external/rl_weights.json(每局最新)与 online_ckpt
快照,从不写;复制自 gomoku/,原始代码不受影响。

用法: python vizplay/vizplay.py            # 图形界面
      VIZ_HEADLESS=1 python vizplay/vizplay.py   # 只跑思考计算冒烟
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox as tkmb

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent))   # 先放主项目(供 external 资源引用)
sys.path.insert(0, str(APP))         # 副本优先:gomoku.* 一律用 vizplay 拷贝

from gomoku.board import BLACK, WHITE, Board, opponent  # noqa: E402
from gomoku.agent import Agent  # noqa: E402
from gomoku.frontier_agent import FrontierAgent  # noqa: E402
from gomoku.layered_agent import LayeredAgent  # noqa: E402
from gomoku.relations import film_values  # noqa: E402
from gomoku.rl_agent import RlAgent  # noqa: E402
from gomoku.spec_agent import SpecAgent  # noqa: E402

sys.path.insert(0, str(APP.parent))
from rl2.core import R2Search, cells_of  # noqa: E402

ROOT = APP.parent  # gobang_algorithm
WEIGHTS = ROOT / "external" / "rl_weights.json"
CKPT = ROOT / "external" / "online_ckpt"
SIZE = 15
CELL = 36            # 逻辑格距(棋盘逻辑宽 = CELL*(SIZE-1) = 504)
LOG_MAR = CELL       # 逻辑边距
LOG_W = LOG_MAR * 2 + CELL * (SIZE - 1)   # 逻辑画布 576x576
HEADLESS = os.environ.get("VIZ_HEADLESS") == "1"

STONE = {BLACK: "#111827", WHITE: "#f8fafc"}
EDGE = {BLACK: "#374151", WHITE: "#94a3b8"}

# 对外展示名(界面与面板都用;内部开发代号一律不出现)
def public_name(eng: str) -> str:
    if eng in PUBLIC_NAMES:
        return PUBLIC_NAMES[eng]
    if eng.startswith("rl-") and eng.endswith("_pre"):
        return f"RL1.0 历史轮 {eng[3:-4]}"     # rl-r32_pre → r32
    return eng


PUBLIC_NAMES = {
    "rl2v": "底片引擎·冠军版",
    "rl": "RL1.0 实时参数",
    "rl-rl1_last_max": "RL1.0 曾胜网页版·最强",
    "frontier": "对照·首端推理版",
    "layered": "对照·预测层版",
    "spec": "对照·忠实版",
    "boosted": "对照·增强版",
}
COL_OWN = (59, 130, 246)      # 我方底片蓝
COL_THEIR = (239, 68, 68)     # 敌方底片红
COL_SHARED = (168, 85, 247)   # 敌我重合(第三情况)紫
BG = "#0b1220"
BOARD = "#f2e0b4"             # 棋盘米黄(木纹板)
GRID = "#8a5a2b"              # 网格深棕(米黄板上)
LAST = "#b45309"              # 最后一手标记(深橙,米黄板上醒目)


def _dpi():
    """Windows 高 DPI 感知:tk 坐标与物理像素一致(防点击偏移)。"""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def make_engine(kind: str, player: int, window: int = 9):
    """引擎工厂;kind: rl2v / rl-rl1_last_max / rl-<stem> / frontier 等。"""
    if kind in ("rl2", "rl2c", "rl2v"):
        # 公开版:冠军版(rl2v)与历史开发名(rl2/rl2c)共用同一实现
        return RlAgent(player, weights_path=str(WEIGHTS))
    cls_map = {"frontier": FrontierAgent, "layered": LayeredAgent,
               "spec": SpecAgent, "boosted": Agent, "rl": RlAgent}
    if kind.startswith("rl-"):
        stem = kind[3:]
        path = CKPT / f"{stem}.json"
        return RlAgent(player, weights_path=str(path))
    if kind == "rl":
        if os.environ.get("VIZ_FAST") == "1":
            # 加速档(复现搜索用):压低深推预算,单步 ~2-5s
            return RlAgent(player, weights_path=str(WEIGHTS),
                           deep_budget=int(os.environ.get("VIZ_DEEP",
                                                          2500)),
                           chain_nodes=int(os.environ.get("VIZ_CHAIN",
                                                          400)),
                           deep_leaf_nodes=1500)
        return RlAgent(player, weights_path=str(WEIGHTS))
    return cls_map[kind](player, 4, window)


def rl_versions():
    """版本列表(对外名):[实时权重 + 曾胜最强 + 各轮快照(新→旧)],全蓝标家族。"""
    out = []
    try:
        d = json.loads(WEIGHTS.read_text(encoding="utf-8"))
        n = len(d.get("geo_patches", []) or [])
        out.append(("rl", f"RL1.0 实时参数 ({n}槽)"))
    except Exception:
        out.append(("rl", "RL1.0 实时参数"))
    seen = set()
    paths = list(CKPT.glob("r*_pre.json"))
    special = CKPT / "rl1_last_max.json"
    if special.exists():
        paths.append(special)
    for p in sorted(paths, key=lambda x: str(x), reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        gp = d.get("geo_patches", []) or []
        n = len(gp)
        if n == 0:
            continue
        sig = frozenset((int(e["sig"][0] if isinstance(e.get("sig"),
                                                       list) else e["sig"]),
                         round(float(e.get("w", 0.0)), 4)) for e in gp)
        if sig in seen:
            continue
        seen.add(sig)
        stem = p.stem
        if stem == "rl1_last_max":
            continue                       # 曾胜置顶统一插入
        out.append((f"rl-{stem}", f"RL1.0 历史轮 {stem[:-4]}" if
                    stem.endswith("_pre") else f"RL1.0 {stem}"))
    sp = CKPT / "rl1_last_max.json"
    if sp.exists():
        try:
            d = json.loads(sp.read_text(encoding="utf-8"))
            n = len(d.get("geo_patches", []) or [])
            out.insert(1, ("rl-rl1_last_max",
                           f"RL1.0 曾胜网页版·最强 ({n}槽)"))
        except Exception:
            pass
    return out


RL2_PATH = ROOT / "external" / "rl2_params.json"


def _rl2_params():
    """最强参数:每次实时读文件(重训即生效,不缓存)。"""
    try:
        return json.loads(RL2_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def r2_reasoning(board: Board, player: int) -> dict:
    """最强(v3.3 参数)推理:层集/视图/值 → 可视化数据。"""
    try:
        s = R2Search(board, player, params=_rl2_params())
        levels, _rec = s.run()
        out = []
        vals = [lv["value"] for lv in levels]
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        rng = (hi - lo) or 1.0
        for lv in levels[:16]:
            node = s.by_key.get(lv["key"])
            if node is None:
                continue
            cells = []
            for f in node.films:
                cells.extend([list(c) for c in cells_of(f)])
            out.append({"depth": lv["depth"], "view": node.view,
                        "gkey": lv.get("gkey", node.gkey),
                        "value": lv["value"],
                        "norm": (lv["value"] - lo) / rng,
                        "cells": cells})
        return {"levels": out}
    except Exception:
        return {"levels": []}


def choose_rl2(board: Board, player: int, engine, mv):
    """最强(v3.3)走子:基座 RL 出候选与审计;score 层叠加强推理线性分。

    每候选格加分 = Σ(其落入层级的 训练分 s1(view,gkey)+s2(view,depth),
    只算 depth<=3)——与格级 ranking 训练同函数。证明/链层保持基座。
    """
    aud = getattr(engine, "last_audit", None) or {}
    if aud.get("stage") != "score" or not aud.get("scored"):
        return list(mv)
    r2 = r2_reasoning(board, player)
    from rl2.core import _P
    p = _P(_rl2_params())
    gamma = float(os.environ.get("VIZ_GAMMA", "1.0"))
    bonus: dict = {}
    for lv in r2.get("levels", []):
        if lv["depth"] > 3:
            continue
        w = (p.s1(lv["view"], lv.get("gkey") or lv["key"]) +
             p.s2(lv["view"], lv["depth"])) * gamma
        for (r, c) in lv["cells"]:
            bonus[(r, c)] = bonus.get((r, c), 0.0) + w
    best_c, best_s = None, None
    for (s, c) in aud["scored"]:
        total = float(s) + bonus.get(tuple(c), 0.0)
        if best_s is None or total > best_s + 1e-9:
            best_s, best_c = total, c
    return best_c if best_c is not None else list(mv)


def think(board: Board, player: int, kind: str) -> dict:
    """思考数据:先真正决策(任何局面都有落点),再取考虑的底片/候选概率。"""
    from opening_book import book_move

    if kind in ("rl2", "rl2c", "rl2v"):
        bm = book_move(board, player)      # 开局冠军表(4x4 折叠域内)
        if bm is not None:
            view = {"engine": kind, "mv": list(bm), "films": [],
                    "moves": [], "book": True}
            view["r2"] = r2_reasoning(board, player)
            return view
        engine = make_engine("rl", player)
        mv = engine.choose(board)
        mv = choose_rl2(board, player, engine, mv)
    else:
        engine = make_engine(kind, player)
        mv = engine.choose(board)          # 真正决策(空盘=开局中心)
    rel = getattr(engine, "rel_weights", None) or {}
    vals = film_values(board, player, rel)
    films = []
    for v, f in vals:
        cells = sorted(set(tuple(c) for c in f.squares) |
                       set(tuple(c) for c in f.kill))
        films.append({"owner": f.owner, "live": f.is_live, "k": f.k,
                      "cls": f.cls, "v": round(float(v), 3),
                      "cells": [[r, c] for r, c in cells]})
    view = {"engine": kind, "mv": list(mv), "films": films[:14],
            "moves": []}
    aud = getattr(engine, "last_audit", None) or {}
    scored = aud.get("scored") if isinstance(aud, dict) else None
    if scored:
        ss = [float(s) for s, _c in scored]
        m = max(ss)
        # 显示用概率:温度 T=6 摊开(原始分差太大,raw softmax 全是 0/1,
        # 深浅无法分辨;T 只影响显示,不改决策)
        T = 6.0
        ex = [math.exp((s - m) / T) for s in ss]
        tot = sum(ex)
        probs = [e / tot for e in ex]
        for (s, c), p in zip(scored, probs):
            view["moves"].append({"cell": c, "p": round(p, 4),
                                  "s": round(float(s), 2)})
        # 底片采纳强度 = 触及它的候选动作概率最大值
        for f, p in zip(vals, view["films"]):
            s = set(f[1].squares) | set(f[1].kill)
            p["intensity"] = max([pb for (c, pb) in
                                  zip([tuple(c) for _s, c in scored],
                                      probs) if c in s] or [0.0])
    else:
        for f, p in zip(vals, view["films"]):
            p["intensity"] = None          # 非评分路径:按 V 相对深浅
        vmax = max([p["v"] for p in view["films"]] + [1.0])
        if vmax <= 0:
            vmax = 1.0                     # 无正 V(如开局)时全部用底色
        for p in view["films"]:
            p["intensity"] = max(0.0, min(1.0, max(0.0, p["v"]) / vmax))
    if kind.startswith("rl"):
        view["r2"] = r2_reasoning(board, player)
    return view

def blend(rgb, frac):
    """颜色深浅:frac 0..1 → 与米黄棋盘混合的色(模拟透明度)。"""
    bg = (242, 224, 180)
    c = tuple(int(bg[i] + (rgb[i] - bg[i]) * frac) for i in range(3))
    return "#%02x%02x%02x" % c


class App:
    """窗口:顶部控制条 + 左:可拉伸棋盘画布 + 右:信息面板。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("五子棋思考可视化 · 底片引擎")
        root.geometry("1080x760")
        root.minsize(700, 560)
        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=6, pady=(6, 2))
        self.var_mode = tk.StringVar(value="auto")
        for text, val in (("人执黑", "human_b"), ("人执白", "human_w"),
                          ("观战", "auto")):
            tk.Radiobutton(top, text=text, value=val, bg=BG, fg="#cbd5e1",
                           selectcolor="#1e293b", variable=self.var_mode,
                           command=self.reset).pack(side="left")
        vers = [("rl2v", "★ 底片引擎·冠军版(最强·默认)")] + rl_versions()
        names = [("frontier", "对照·首端推理版"),
                 ("layered", "对照·预测层版"),
                 ("spec", "对照·忠实版"),
                 ("boosted", "对照·增强版")]
        tk.Label(top, text="引擎A(黑):", bg=BG, fg="#94a3b8").pack(
            side="left", padx=(10, 0))
        self.var_b = tk.StringVar(value="rl2v")
        self.sel_b = tk.OptionMenu(top, self.var_b, *[v for v, _n in
                                                      vers + names])
        self.sel_b.config(bg="#1e293b", fg="#e2e8f0", highlightthickness=0)
        self.sel_b.pack(side="left")
        self.var_b.trace_add("write", lambda *_: self._tint(self.sel_b,
                                                            self.var_b))
        self._tint(self.sel_b, self.var_b)
        tk.Label(top, text="引擎B(白):", bg=BG, fg="#94a3b8").pack(
            side="left", padx=(10, 0))
        self.var_w = tk.StringVar(value="rl-rl1_last_max")
        self.sel_w = tk.OptionMenu(top, self.var_w, *[v for v, _n in
                                                      vers + names])
        self.sel_w.config(bg="#1e293b", fg="#e2e8f0", highlightthickness=0)
        self.sel_w.pack(side="left")
        self.var_w.trace_add("write", lambda *_: self._tint(self.sel_w,
                                                            self.var_w))
        self._tint(self.sel_w, self.var_w)
        self.var_show = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="显示思考范围", variable=self.var_show,
                       bg=BG, fg="#cbd5e1", selectcolor="#1e293b",
                       command=self.redraw).pack(side="left", padx=10)
        tk.Button(top, text="走一步", command=self.step, bg="#2563eb",
                  fg="white", relief="flat").pack(side="left", padx=4)
        tk.Button(top, text="新对局", command=self.reset, bg="#475569",
                  fg="white", relief="flat").pack(side="left", padx=4)
        tk.Button(top, text="窗口拉大=棋盘变大", command=lambda: None,
                  bg="#1e293b", fg="#64748b", relief="flat").pack(
            side="left", padx=8)
        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True)
        self.cv = tk.Canvas(body, bg=BG, highlightthickness=0)
        self.cv.pack(side="left", fill="both", expand=True)
        self.panel = tk.Frame(body, bg="#0f172a", width=280)
        self.panel.pack(side="right", fill="y")
        self.info = tk.Label(self.panel, justify="left", anchor="nw",
                             bg="#0f172a", fg="#94a3b8",
                             font=("Consolas", 9), wraplength=264)
        self.info.pack(fill="both", expand=True, padx=8, pady=6)
        self.board = Board()
        self.cur = BLACK
        self.view = None
        self._u = 1.0                      # 屏幕缩放(逻辑→像素)
        self._ox = 0.0
        self._oy = 0.0
        self._redraw_pending = False
        self.cv.bind("<Button-1>", self.on_click)
        self.cv.bind("<Configure>", self._on_resize)
        self.reset()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    # ---- 坐标引擎(绘制与点击共用,任何窗口尺寸都一致)----
    def _layout(self):
        w = self.cv.winfo_width()
        h = self.cv.winfo_height()
        if w < 40 or h < 40:               # 首帧未布局时用默认比例
            w, h = 620, 620
        u = min((w - 20) / LOG_W, (h - 20) / LOG_W)
        if u <= 0:
            u = 1.0
        self._u = u
        self._ox = (w - LOG_W * u) / 2.0
        self._oy = (h - LOG_W * u) / 2.0

    def P(self, x, y):
        """逻辑坐标 (x, y) → 屏幕坐标。"""
        return self._ox + x * self._u, self._oy + y * self._u

    def rc_at(self, sx, sy):
        """屏幕坐标 → 棋盘 (r, c);不在盘内返回 None。"""
        px = (sx - self._ox) / self._u - LOG_MAR
        py = (sy - self._oy) / self._u - LOG_MAR
        c = round(px / CELL)
        r = round(py / CELL)
        if 0 <= r < SIZE and 0 <= c < SIZE:
            return r, c
        return None

    def _on_resize(self, _ev=None):
        if self._redraw_pending:
            return
        self._redraw_pending = True

        def go():
            self._redraw_pending = False
            self.redraw()

        self.root.after(40, go)            # 合并连续缩放事件

    @staticmethod
    def _tint(sel, var):
        """引擎标色:紫=冠军版;蓝=RL1.0 家族(实时/曾胜/历史轮);灰=对照。"""
        v = var.get()
        if v == "rl2v":
            sel.config(bg="#581c87", fg="#f3e8ff",
                       activebackground="#6d28d9",
                       activeforeground="#fff")
        elif v == "rl" or v.startswith("rl-"):
            sel.config(bg="#1e3a8a", fg="#e2e8f0",
                       activebackground="#1d4ed8")
        else:
            sel.config(bg="#1e293b", fg="#e2e8f0",
                       activebackground="#334155")

    # ---- 对局 ----
    def reset(self):
        self.board = Board()
        self.cur = BLACK
        self.view = None
        self.winner = None          # 终局标记:None=进行中,1/2=胜方,0=和
        self.redraw()
        self.set_info("新对局。点棋盘落子;观战模式点『走一步』看引擎行棋。\n"
                      "拖动窗口右下角可放大棋盘。")

    def _finish(self, winner, msg):
        """棋局结束:标记胜方、提示、弹窗询问是否再来一局。"""
        self.winner = winner
        self.set_info(msg)
        self.redraw()
        again = tkmb.askyesno(
            "本局结束",
            ("再来一局?" if winner == 0 else
             f"{'黑方' if winner == BLACK else '白方'}获胜!\n再来一局?"))
        if again:
            self.reset()

    def _after_move(self, player, extra=""):
        """每次落子后的终局检查;未终局返回 False。"""
        if self.board.is_win_move(self.board.moves[-1][0],
                                  self.board.moves[-1][1], player):
            who = "黑方" if player == BLACK else "白方"
            self._finish(player, f"{who}胜!{extra}")
            return True
        if self.board.full():
            self._finish(0, "棋盘已满,和棋。")
            return True
        return False

    def step(self):
        if self.winner is not None:
            return
        mode = self.var_mode.get()
        if mode == "auto":
            if self.board.full():
                return
            kind = self.var_b.get() if self.cur == BLACK else \
                self.var_w.get()
            self.ai_move(self.cur, kind)
        elif mode == "human_b":
            if self.cur == WHITE:
                self.ai_move(WHITE, self.var_w.get())
        elif mode == "human_w":
            if self.cur == BLACK:
                self.ai_move(BLACK, self.var_b.get())

    def ai_move(self, player, kind):
        if self.winner is not None:
            return
        self.set_info(f"{'黑' if player == BLACK else '白'}方思考中…")
        self.root.update()
        t0 = time.perf_counter()
        print(f"[viz] {kind} {'黑' if player == BLACK else '白'}思考开始",
              flush=True)
        try:
            view = think(self.board, player, kind)
            print(f"[viz] 思考完成 {time.perf_counter()-t0:.1f}s "
                  f"落点 {view['mv']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.set_info(f"引擎异常: {exc!r}")
            return
        r, c = view["mv"]
        self.board.place(r, c, player)
        self.view = view if self.cur == player else self.view
        self.set_info(self.describe(view))
        self.cur = opponent(player)
        self.redraw()
        self._after_move(player, "思考可视化已展示推理过程。")

    def on_click(self, ev):
        if self.winner is not None:
            return
        mode = self.var_mode.get()
        if mode == "auto":
            return
        me = BLACK if mode == "human_b" else WHITE
        if self.cur != me:
            return
        rc = self.rc_at(ev.x, ev.y)
        if rc is None:
            return
        r, c = rc
        if self.board.get(r, c) != 0:
            return
        self.board.place(r, c, me)
        self.cur = opponent(me)
        self.redraw()
        if self._after_move(me, "思考可视化已展示推理过程。"):
            return
        self.step()

    def describe(self, view) -> str:
        eng = view["engine"]
        name = public_name(eng)
        if eng.startswith("rl"):
            n = len(view["films"])
            top = view["films"][:6]
            lines = [f"{name} | 候选 {len(view['moves'])} | "
                     f"底片 {n}"]
            r2 = view.get("r2") or {}
            if r2.get("levels"):
                from collections import Counter
                cnt = Counter(lv["view"] for lv in r2["levels"])
                lines[0] += (f" | 最强层 {len(r2['levels'])}"
                             f"(我{cnt.get('my',0)}/"
                             f"敌{cnt.get('enemy',0)}/"
                             f"合{cnt.get('shared',0)})")
                best = max(r2["levels"], key=lambda x: x["value"])
                lines.append(f"最强高值层: d{best['depth']}"
                             f"{best['view']} val{best['value']:.2f}")
            lines.append("候选采纳概率 top5:")
            for m in view["moves"][:5]:
                lines.append(f"  ({m['cell'][0]:2d},{m['cell'][1]:2d}) "
                             f"{m['p'] * 100:5.1f}%")
            lines.append("考虑底片(V/采纳):")
            for f in top:
                who = "我" if f["owner"] == BLACK else "敌"
                liv = "活" if f["live"] else "潜"
                it = f["intensity"]
                itx = "-" if it is None else f"{it * 100:4.0f}%"
                lines.append(f"  {who}{liv}k{f['k']} V={f['v']:6.2f} "
                             f"采纳{itx}")
            return "\n".join(lines)
        return (f"{name} 已落子 {view['mv']}\n"
                "(非 RL 引擎无内部采纳概率信号,\n思考深浅只对 RL 引擎绘制)")

    def set_info(self, text):
        self.info.config(text=text)

    # ---- 画板(全部经 _layout/P 变换,随窗口缩放)----
    def redraw(self):
        self._layout()
        u = self._u
        cv = self.cv
        cv.delete("all")
        fs = max(8, int(11 * u / 1.2))
        # 米黄棋盘板(网格区外扩一圈;坐标字留在板外深色区)
        bx0, by0 = self.P(LOG_MAR - 10, LOG_MAR - 10)
        bx1, by1 = self.P(LOG_MAR + CELL * (SIZE - 1) + 10,
                          LOG_MAR + CELL * (SIZE - 1) + 10)
        cv.create_rectangle(bx0, by0, bx1, by1, fill=BOARD,
                            outline="#7a4a1e", width=int(max(1, 1.1 * u)))
        # 网格(逻辑交点 LOG_MAR + i*CELL)
        for i in range(SIZE):
            x = LOG_MAR + i * CELL
            x1, y1 = self.P(x, LOG_MAR)
            x2, y2 = self.P(x, LOG_MAR + CELL * (SIZE - 1))
            cv.create_line(x1, y1, x2, y2, fill=GRID)
            x1, y1 = self.P(LOG_MAR, x)
            x2, y2 = self.P(LOG_MAR + CELL * (SIZE - 1), x)
            cv.create_line(x1, y1, x2, y2, fill=GRID)
        # 坐标字
        for i in range(SIZE):
            x, y = self.P(LOG_MAR + i * CELL, LOG_MAR - 18)
            cv.create_text(x, y, text=chr(65 + i), fill="#475569",
                           font=("Consolas", fs))
            x, y = self.P(LOG_MAR - 20, LOG_MAR + i * CELL)
            cv.create_text(x, y, text=str(SIZE - i), fill="#475569",
                           font=("Consolas", fs))
        board = self.board
        grid = board.grid
        pad = 0.36 * CELL * u              # 色块/棋子相对逻辑 CELL 缩放
        # 思考范围(只在 RL 引擎下有真实的"采纳概率"内部信号)
        if self.var_show.get() and self.view \
                and self.view["engine"].startswith("rl"):
            for f in self.view["films"]:
                owner = f["owner"]
                live = f["live"]
                it = f["intensity"]
                frac = (0.18 + 0.82 * it) if it is not None else 0.30
                rgb = COL_OWN if owner == BLACK else COL_THEIR
                for (r, c) in f["cells"]:
                    x0, y0 = self.P(LOG_MAR + c * CELL,
                                    LOG_MAR + r * CELL)
                    if live:
                        cv.create_rectangle(x0 - pad, y0 - pad,
                                            x0 + pad, y0 + pad,
                                            fill=blend(rgb, frac),
                                            outline="")
                    else:
                        cv.create_rectangle(x0 - pad, y0 - pad,
                                            x0 + pad, y0 + pad,
                                            outline=blend(
                                                rgb, min(1.0, frac + .3)),
                                            dash=(2, 2))
        # 最强(v3.3)推理层集:三视图色,深浅=层值(在底片轮廓之下、棋子之上)
        r2 = self.view.get("r2") if isinstance(self.view, dict) else None
        if self.var_show.get() and r2 and r2.get("levels") \
                and self.view["engine"].startswith("rl"):
            vc = {"my": COL_OWN, "enemy": COL_THEIR, "shared": COL_SHARED}
            for lv in r2["levels"][:14]:
                if lv["depth"] > 3:
                    continue
                rgb = vc.get(lv["view"], COL_SHARED)
                frac = 0.12 + 0.55 * max(0.0, min(1.0, lv["norm"]))
                s2 = 0.24 * CELL * u
                for (r, c) in lv["cells"][:40]:
                    x0, y0 = self.P(LOG_MAR + c * CELL,
                                    LOG_MAR + r * CELL)
                    cv.create_rectangle(x0 - s2, y0 - s2, x0 + s2,
                                        y0 + s2, fill=blend(rgb, frac),
                                        outline="")
        # 棋子
        sr = 0.42 * CELL * u
        for r in range(SIZE):
            for c in range(SIZE):
                v = grid[r][c]
                if not v:
                    continue
                x, y = self.P(LOG_MAR + c * CELL, LOG_MAR + r * CELL)
                cv.create_oval(x - sr, y - sr, x + sr, y + sr,
                               fill=STONE[v], outline=EDGE[v], width=1)
        # 最后一手标记
        last = {}
        for (r, c, p) in board.moves:
            last[p] = (r, c)
        for p, (r, c) in last.items():
            x, y = self.P(LOG_MAR + c * CELL, LOG_MAR + r * CELL)
            cv.create_oval(x - 4, y - 4, x + 4, y + 4, outline=LAST,
                           width=int(max(1, 1.6 * u)))


def headless_smoke():
    """无头冒烟:空盘第一步 + 真实前缀思考 + 坐标往返。"""
    app_probe = App.__new__(App)           # 只用几何换算,不开窗口
    app_probe._u = 1.3
    app_probe._ox = (900 - LOG_W * 1.3) / 2
    app_probe._oy = (700 - LOG_W * 1.3) / 2
    bad = 0
    for r in range(SIZE):
        for c in range(SIZE):
            x, y = app_probe.P(LOG_MAR + c * CELL, LOG_MAR + r * CELL)
            if app_probe.rc_at(x, y) != (r, c):
                bad += 1
    print(f"坐标往返校验: 225 点误差 {bad}")
    for kind in ("rl", "frontier"):
        b = Board()
        view = think(b, BLACK, kind)
        assert "mv" in view and view["mv"], (kind, view)
        print(f"空盘 {kind}: 落点 {view['mv']} 底片 {len(view['films'])}",
              flush=True)
    files = sorted((ROOT / "external" / "online").glob("moves_*.json"))
    if files:
        fp = files[-1]
        data = json.loads(fp.read_text(encoding="utf-8"))
        gid = int(fp.stem.split("_")[1])
        moves = data["games"][str(gid)]
        rows = [x for x in data.get("rows", []) if x.get("stage") == "score"]
        row = rows[len(rows) // 2]
        b = Board()
        for (r, c, p) in moves[: row["ply"]]:
            b.place(r, c, p)
        view = think(b, row["cur"], "rl")
        print(json.dumps({
            "gid": gid, "ply": row["ply"], "mv": view["mv"],
            "n_films": len(view["films"]),
            "n_moves": len(view["moves"]),
            "top_moves": view["moves"][:3]}, ensure_ascii=False, indent=1))
    else:
        print("(公开版无 external/online 对局谱,跳过前缀复盘冒烟)",
              flush=True)


def main():
    if HEADLESS:
        headless_smoke()
        return
    _dpi()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
