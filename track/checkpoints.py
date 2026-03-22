# -*- coding: utf-8 -*-
"""
检查点门线：左边界到右边界线段；按序通过加分、乱序扣分；带冷却防抖。
按序通过最后一道门计为一整圈，额外 lap_bonus；PSO 回合结束若未撞墙却未完成至少一圈则扣 no_lap_penalty。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


def unpack_track(track):
    """
    兼容三元组 (c, l, r) 与四元组 (c, l, r, checkpoint_gates)。
    返回 (centerline, left_bound, right_bound, gates)，gates 可为 None 或 (K,2,2) 数组。
    """
    if track is None or len(track) < 3:
        raise ValueError("track 至少需要 centerline, left_bound, right_bound")
    c, l, r = track[0], track[1], track[2]
    gates = track[3] if len(track) >= 4 else None
    if gates is not None and getattr(gates, "size", 0) == 0:
        gates = None
    return c, l, r, gates


def segment_crosses_segment(p0, p1, q0, q1, eps=1e-10):
    """
    判断二维线段 (p0,p1) 与 (q0,q1) 是否相交（含端点接触）。
    参数方程 + 叉积；平行或共线时若重叠则视为相交。
    """
    p0 = np.asarray(p0, dtype=np.float64).ravel()[:2]
    p1 = np.asarray(p1, dtype=np.float64).ravel()[:2]
    q0 = np.asarray(q0, dtype=np.float64).ravel()[:2]
    q1 = np.asarray(q1, dtype=np.float64).ravel()[:2]
    r = p1 - p0
    s = q1 - q0
    rxs = r[0] * s[1] - r[1] * s[0]
    qmp = q0 - p0
    qpxr = qmp[0] * r[1] - qmp[1] * r[0]

    if abs(rxs) < eps and abs(qpxr) < eps:
        # 共线：判断投影区间是否重叠
        if abs(r[0]) >= abs(r[1]) + eps:
            t0 = (q0[0] - p0[0]) / (r[0] + eps)
            t1 = (q1[0] - p0[0]) / (r[0] + eps)
        else:
            t0 = (q0[1] - p0[1]) / (r[1] + eps)
            t1 = (q1[1] - p0[1]) / (r[1] + eps)
        t0, t1 = min(t0, t1), max(t0, t1)
        return not (t1 < 0 or t0 > 1)

    if abs(rxs) < eps:
        return False

    t = (qmp[0] * s[1] - qmp[1] * s[0]) / rxs
    u = qpxr / rxs
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


@dataclass
class CheckpointState:
    """检查点计分状态：下一个期望门下标、每门冷却、已完成整圈次数。"""

    next_idx: int = 0
    cooldown: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    laps_completed: int = 0

    @classmethod
    def empty(cls, n_gates: int):
        return cls(
            next_idx=0,
            cooldown=np.zeros(max(0, n_gates), dtype=np.int32),
            laps_completed=0,
        )


def score_checkpoint_crossing(
    prev_xy, curr_xy, gates, state, bonus, penalty, cooldown_steps, lap_bonus=0.0
):
    """
    本步位移线段与门线求交，更新得分与状态。

    Parameters
    ----------
    prev_xy, curr_xy : array-like (2,)
    gates : (K, 2, 2) 或 None
    state : CheckpointState
    bonus, penalty : float
    cooldown_steps : int
    lap_bonus : float
        通过最后一道门（闭环一整圈）时额外加分。

    Returns
    -------
    delta_score : float
    state : CheckpointState（原地更新 cooldown / next_idx）
    """
    if gates is None or len(gates) == 0:
        return 0.0, state

    K = len(gates)
    if state.cooldown.shape[0] != K:
        state = CheckpointState.empty(K)

    state.cooldown = np.maximum(0, state.cooldown - 1)

    hits = []
    for k in range(K):
        if state.cooldown[k] > 0:
            continue
        q0, q1 = gates[k, 0], gates[k, 1]
        if segment_crosses_segment(prev_xy, curr_xy, q0, q1):
            hits.append(k)

    delta = 0.0
    if not hits:
        return delta, state

    if state.next_idx in hits:
        old_next = state.next_idx
        delta += bonus
        # 通过最后一道门后 next 回到 0，记为一整圈
        if old_next == K - 1:
            state.laps_completed += 1
            delta += float(lap_bonus)
        state.cooldown[old_next] = int(cooldown_steps)
        state.next_idx = (old_next + 1) % K
    else:
        delta -= penalty
        for k in hits:
            state.cooldown[k] = int(cooldown_steps)

    return delta, state


def finalize_checkpoint_episode_score(
    checkpoint_score, gates, laps_completed, no_lap_penalty, collision=False
):
    """
    回合结束：若赛道上有关卡、本回合未撞墙、但未完成至少一整圈，扣 no_lap_penalty。
    撞墙时不扣（适应度已按碰撞归零）。无检查点时不改分。
    """
    if gates is None or len(gates) == 0:
        return float(checkpoint_score)
    if collision:
        return float(checkpoint_score)
    pen = float(no_lap_penalty or 0.0)
    if pen != 0.0 and laps_completed < 1:
        return float(checkpoint_score) - pen
    return float(checkpoint_score)


def load_track_npz(path):
    """
    从 PSO/主流程保存的 training_track.npz 加载赛道，返回结构与 generate_track 一致：
    (centerline, left_bound, right_bound, checkpoint_gates)。
    用于神经网络测试与 PSO 使用同一条赛道（同一随机种子生成的几何存档）。
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到赛道存档: {path}")
    data = np.load(path, allow_pickle=False)
    centerline = np.asarray(data["centerline"], dtype=np.float64)
    left_bound = np.asarray(data["left_bound"], dtype=np.float64)
    right_bound = np.asarray(data["right_bound"], dtype=np.float64)
    if "checkpoint_gates" in data.files:
        checkpoint_gates = np.asarray(data["checkpoint_gates"], dtype=np.float64)
        if checkpoint_gates.size == 0:
            checkpoint_gates = np.zeros((0, 2, 2), dtype=np.float64)
    else:
        checkpoint_gates = np.zeros((0, 2, 2), dtype=np.float64)
    return (centerline, left_bound, right_bound, checkpoint_gates)
