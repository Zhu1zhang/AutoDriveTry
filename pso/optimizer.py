# -*- coding: utf-8 -*-
"""
粒子群优化 (PSO) 搜索最优驾驶策略。
粒子为策略参数向量，适应度由离边界距离、速度、碰撞、轨迹平滑等组成。
"""
import numpy as np

import config
from sensor.radar import get_radar_distances


def _params_to_steer_speed(params, radar_distances, prev_steer=0.0):
    """
    将 6 维策略参数与当前雷达距离映射为转向角与目标速度。
    左前/右前雷达加权差决定转向，最近距离与速度系数决定目标速度。
    """
    # 左前约 index 2~5，右前约 10~13（0=车头，逆时针）
    left_indices = list(range(2, 6))
    right_indices = list(range(10, 14))
    left_val = np.mean(radar_distances[left_indices]) + 1e-6
    right_val = np.mean(radar_distances[right_indices]) + 1e-6
    k_steer_left, k_steer_right, k_speed, bias_steer, min_dist_weight, steer_smooth = params
    # 左近则向右转（正 steer），右近则向左转（负 steer）
    steer = k_steer_left * (1.0 / left_val) - k_steer_right * (1.0 / right_val) + bias_steer
    steer = steer + steer_smooth * prev_steer
    steer = np.clip(steer, -config.CAR["max_steer"], config.CAR["max_steer"])

    min_d = np.min(radar_distances) + 1e-6
    # 目标速度：越近墙越慢
    target_speed = k_speed * min_d * min_dist_weight + config.CAR["min_speed"] * (1 - min_dist_weight)
    target_speed = np.clip(target_speed, config.CAR["min_speed"], config.CAR["max_speed"])
    return steer, target_speed


def _check_collision(pos, centerline, left_bound, right_bound, margin=3.0):
    """
    判断车辆是否撞墙：若到中心线最近点的距离大于半宽（即超出左右边界）则碰撞。
    或到左/右边界的最小距离小于 margin 则判为撞墙。
    """
    pos = np.asarray(pos, dtype=float)
    n = len(centerline)
    half_width = config.TRACK["half_width"]
    # 点到中心线各线段的距离，取最小
    min_d_to_center = float("inf")
    for i in range(n):
        a, b = centerline[i], centerline[(i + 1) % n]
        ab = b - a
        ap = pos - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-10)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        d = np.linalg.norm(pos - proj)
        if d < min_d_to_center:
            min_d_to_center = d
    if min_d_to_center > half_width - margin:
        return True
    # 到左右边界的距离若过小也判为撞墙
    for i in range(n):
        a, b = left_bound[i], left_bound[(i + 1) % n]
        ab = b - a
        ap = pos - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-10)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        if np.linalg.norm(pos - proj) < margin:
            return True
        a, b = right_bound[i], right_bound[(i + 1) % n]
        ab = b - a
        ap = pos - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-10)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        if np.linalg.norm(pos - proj) < margin:
            return True
    return False


def _simulate_episode(track, params, max_steps, rng):
    """
    用给定策略参数在赛道上仿真一轮，返回 (collision, total_dist, avg_speed, min_radar, steer_changes)。
    """
    centerline, left_bound, right_bound = track
    n_pts = len(centerline)
    # 起始：中心线第一点，朝向为沿中心线方向
    start = centerline[0]
    diff = centerline[1] - centerline[0]
    start_heading = np.arctan2(diff[1], diff[0])

    x, y = float(start[0]), float(start[1])
    theta = start_heading
    v = config.CAR["min_speed"]
    dt = config.CAR["dt"]
    max_steer = config.CAR["max_steer"]
    wheelbase = config.CAR["wheelbase"]

    total_dist = 0.0
    min_radar = float("inf")
    prev_steer = 0.0
    steer_list = []
    collision = False

    for step in range(max_steps):
        radar = get_radar_distances((x, y), theta, left_bound, right_bound)
        min_radar = min(min_radar, np.min(radar))
        steer, target_speed = _params_to_steer_speed(params, radar, prev_steer)
        steer_list.append(steer)
        prev_steer = steer

        # 自行车模型
        v = v + config.CAR["accel"] * (target_speed - v)
        v = np.clip(v, config.CAR["min_speed"], config.CAR["max_speed"])
        omega = v * np.tan(np.clip(steer, -max_steer, max_steer)) / wheelbase
        theta += omega * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        total_dist += v * dt

        if _check_collision((x, y), centerline, left_bound, right_bound):
            collision = True
            break

    steer_changes = np.sum(np.abs(np.diff(steer_list))) if len(steer_list) > 1 else 0
    avg_speed = total_dist / ((step + 1) * dt) if (step + 1) > 0 else 0
    return collision, total_dist, avg_speed, min_radar, steer_changes


def _simulate_episode_with_path(track, params, max_steps):
    """
    用给定策略仿真一轮，并记录轨迹点，用于可视化。
    返回 (collision, total_dist, avg_speed, min_radar, steer_changes, path_xy)。
    path_xy: list of (x, y)，长度为步数+1（含起点）。
    """
    centerline, left_bound, right_bound = track
    start = centerline[0]
    diff = centerline[1] - centerline[0]
    start_heading = np.arctan2(diff[1], diff[0])
    x, y = float(start[0]), float(start[1])
    theta = start_heading
    v = config.CAR["min_speed"]
    dt = config.CAR["dt"]
    max_steer = config.CAR["max_steer"]
    wheelbase = config.CAR["wheelbase"]
    total_dist = 0.0
    min_radar = float("inf")
    prev_steer = 0.0
    steer_list = []
    path_xy = [(x, y)]
    collision = False
    for step in range(max_steps):
        radar = get_radar_distances((x, y), theta, left_bound, right_bound)
        min_radar = min(min_radar, np.min(radar))
        steer, target_speed = _params_to_steer_speed(params, radar, prev_steer)
        steer_list.append(steer)
        prev_steer = steer
        v = v + config.CAR["accel"] * (target_speed - v)
        v = np.clip(v, config.CAR["min_speed"], config.CAR["max_speed"])
        omega = v * np.tan(np.clip(steer, -max_steer, max_steer)) / wheelbase
        theta += omega * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        total_dist += v * dt
        path_xy.append((x, y))
        if _check_collision((x, y), centerline, left_bound, right_bound):
            collision = True
            break
    steer_changes = np.sum(np.abs(np.diff(steer_list))) if len(steer_list) > 1 else 0
    avg_speed = total_dist / ((step + 1) * dt) if (step + 1) > 0 else 0
    return collision, total_dist, avg_speed, min_radar, steer_changes, path_xy


def _fitness(collision, total_dist, avg_speed, min_radar, steer_changes):
    """适应度：不撞墙前提下，距离远、速度快、轨迹平滑得分高。"""
    if collision:
        return 0.0
    # 正项：行驶距离、平均速度、最小雷达距离
    f = total_dist * 0.5 + avg_speed * 2.0 + min_radar * 0.02
    # 平滑惩罚
    f -= steer_changes * 0.01
    return max(0.0, f)


def run_pso(track, max_steps=None, n_particles=None, n_iters=None):
    """
    在给定赛道上运行 PSO，搜索最优驾驶策略参数。

    Parameters
    ----------
    track : tuple of (centerline, left_bound, right_bound)
    max_steps : int, optional
    n_particles : int, optional
    n_iters : int, optional

    Returns
    -------
    best_params : np.ndarray, shape (6,)
    fitness_history : list of float
    """
    cfg = config.PSO
    max_steps = max_steps or cfg["max_steps_per_episode"]
    n_particles = n_particles or cfg["n_particles"]
    n_iters = n_iters or cfg["n_iters"]
    bounds = np.array(cfg["param_bounds"])
    dim = len(bounds)
    rng = np.random.default_rng(42)

    # 初始化粒子位置与速度
    low = bounds[:, 0]
    high = bounds[:, 1]
    X = rng.uniform(low, high, size=(n_particles, dim))
    V = rng.uniform(-(high - low) * 0.1, (high - low) * 0.1, size=(n_particles, dim))

    pbest = np.copy(X)
    pbest_f = np.full(n_particles, -np.inf)
    gbest = np.copy(X[0])
    gbest_f = -np.inf
    w = cfg["w"]
    c1 = cfg["c1"]
    c2 = cfg["c2"]
    fitness_history = []

    for it in range(n_iters):
        for i in range(n_particles):
            collision, total_dist, avg_speed, min_radar, steer_changes = _simulate_episode(
                track, X[i], max_steps, rng
            )
            f = _fitness(collision, total_dist, avg_speed, min_radar, steer_changes)
            if f > pbest_f[i]:
                pbest_f[i] = f
                pbest[i] = X[i].copy()
            if f > gbest_f:
                gbest_f = f
                gbest = X[i].copy()
        fitness_history.append(gbest_f)
        # 每轮输出进度，避免长时间无输出
        print(f"  PSO 迭代 {it + 1}/{n_iters}  当前最优适应度: {gbest_f:.2f}", flush=True)

        # 更新速度与位置
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        for d in range(dim):
            X[:, d] = np.clip(X[:, d], low[d], high[d])

    return gbest, fitness_history


def plot_pso_results(track, best_params, fitness_history, save_path=None, show=True):
    """
    PSO 结果可视化：适应度曲线 + 最优策略在赛道上的轨迹。

    Parameters
    ----------
    track : tuple of (centerline, left_bound, right_bound)
    best_params : np.ndarray, shape (6,)
    fitness_history : list of float
    save_path : str, optional
        保存图片路径，如 "data/pso_result.png"
    show : bool
        是否弹出显示窗口（默认 True）
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    centerline, left_bound, right_bound = track
    max_steps = config.PSO.get("max_steps_per_episode", 200)
    _, _, _, _, _, path_xy = _simulate_episode_with_path(track, best_params, max_steps)
    path_xy = np.array(path_xy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 左图：适应度随迭代变化
    ax1.plot(fitness_history, color="C0", linewidth=2)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best Fitness")
    ax1.set_title("PSO Fitness Curve")
    ax1.grid(True, alpha=0.3)

    # 右图：赛道 + 最优轨迹
    ax2.plot(centerline[:, 0], centerline[:, 1], "k-", lw=1.5, alpha=0.7, label="Centerline")
    ax2.plot(left_bound[:, 0], left_bound[:, 1], "gray", lw=1, alpha=0.6)
    ax2.plot(right_bound[:, 0], right_bound[:, 1], "gray", lw=1, alpha=0.6)
    ax2.plot(path_xy[:, 0], path_xy[:, 1], "C1-", lw=2, label="Best trajectory")
    ax2.plot(path_xy[0, 0], path_xy[0, 1], "go", markersize=8, label="Start")
    ax2.plot(path_xy[-1, 0], path_xy[-1, 1], "ro", markersize=8, label="End")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Track & Best PSO Trajectory")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"  PSO 可视化已保存: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
