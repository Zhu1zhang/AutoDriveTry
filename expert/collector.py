# -*- coding: utf-8 -*-
"""
专家数据采集：用 PSO 最优策略在赛道上仿真，每帧记录
输入：16 雷达距离 + 当前速度（17 维）
输出：转向角、期望速度（2 维）
保存为 CSV 与 NumPy。
"""
import os
import numpy as np

import config
from sensor.radar import get_radar_distances
from pso.optimizer import _params_to_steer_speed, _check_collision
from track.checkpoints import unpack_track


def _simulate_and_record(track, pso_params, max_frames):
    """
    用 pso_params 在 track 上仿真，每帧记录 (radar_16, speed) -> (steer, target_speed)。
    仅在未碰撞的帧记录。返回 X (N, 17), Y (N, 2)。
    """
    centerline, left_bound, right_bound, _gates = unpack_track(track)
    start = centerline[0]
    diff = centerline[1] - centerline[0]
    start_heading = np.arctan2(diff[1], diff[0])

    x, y = float(start[0]), float(start[1])
    theta = start_heading
    v = config.CAR["min_speed"]
    dt = config.CAR["dt"]
    max_steer = config.CAR["max_steer"]
    wheelbase = config.CAR["wheelbase"]

    rows = []
    prev_steer = 0.0

    for step in range(max_frames):
        radar = get_radar_distances((x, y), theta, left_bound, right_bound)
        steer, target_speed = _params_to_steer_speed(pso_params, radar, prev_steer)
        prev_steer = steer

        # 输入 17 维，输出 2 维
        row_x = list(radar) + [v]
        row_y = [steer, target_speed]
        rows.append((row_x, row_y))

        # 更新状态
        v = v + config.CAR["accel"] * (target_speed - v)
        v = np.clip(v, config.CAR["min_speed"], config.CAR["max_speed"])
        omega = v * np.tan(np.clip(steer, -max_steer, max_steer)) / wheelbase
        theta += omega * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt

        if _check_collision((x, y), centerline, left_bound, right_bound):
            break

    if not rows:
        return np.zeros((0, 17)), np.zeros((0, 2))

    X = np.array([r[0] for r in rows], dtype=np.float64)
    Y = np.array([r[1] for r in rows], dtype=np.float64)
    return X, Y


def collect_expert_data(track, pso_params, output_dir, max_frames=None):
    """
    采集专家数据并保存到 output_dir。

    Parameters
    ----------
    track : tuple of (centerline, left_bound, right_bound)
    pso_params : np.ndarray, shape (6,)
    output_dir : str
        输出目录，将写入 expert_data.csv、expert_X.npy、expert_Y.npy。
    max_frames : int, optional
    """
    cfg = config.EXPERT
    max_frames = max_frames or cfg["max_frames"]
    os.makedirs(output_dir, exist_ok=True)

    X, Y = _simulate_and_record(track, pso_params, max_frames)

    if cfg["output_npy"]:
        np.save(os.path.join(output_dir, "expert_X.npy"), X)
        np.save(os.path.join(output_dir, "expert_Y.npy"), Y)

    if cfg["output_csv"]:
        import csv
        csv_path = os.path.join(output_dir, "expert_data.csv")
        headers = [f"radar_{i}" for i in range(16)] + ["speed", "steer", "target_speed"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for i in range(len(X)):
                w.writerow(list(X[i]) + list(Y[i]))

    return X, Y
