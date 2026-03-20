# -*- coding: utf-8 -*-
"""
粒子群优化 (PSO) 搜索最优驾驶策略。
粒子为策略参数向量，适应度由离边界距离、速度、碰撞、轨迹平滑等组成。
"""
import numpy as np

import config
from sensor.radar import get_radar_distances
from track.checkpoints import CheckpointState, score_checkpoint_crossing, unpack_track


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
    用给定策略参数在赛道上仿真一轮。
    返回 (collision, total_dist, avg_speed, min_radar, steer_changes, survival_time, checkpoint_score)。
    survival_time = 实际仿真步数 * dt（秒），碰撞提前结束则小于 max_steps*dt。
    """
    centerline, left_bound, right_bound, gates = unpack_track(track)
    cp_cfg = config.CHECKPOINT
    n_gates = len(gates) if gates is not None else 0
    cp_state = CheckpointState.empty(n_gates)
    checkpoint_score = 0.0
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
    step = -1

    for step in range(max_steps):
        radar = get_radar_distances((x, y), theta, left_bound, right_bound)
        min_radar = min(min_radar, np.min(radar))
        steer, target_speed = _params_to_steer_speed(params, radar, prev_steer)
        steer_list.append(steer)
        prev_steer = steer

        px, py = x, y
        # 自行车模型
        v = v + config.CAR["accel"] * (target_speed - v)
        v = np.clip(v, config.CAR["min_speed"], config.CAR["max_speed"])
        omega = v * np.tan(np.clip(steer, -max_steer, max_steer)) / wheelbase
        theta += omega * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        total_dist += v * dt

        d_cp, cp_state = score_checkpoint_crossing(
            (px, py),
            (x, y),
            gates,
            cp_state,
            cp_cfg["pass_bonus"],
            cp_cfg["wrong_penalty"],
            cp_cfg["cooldown_steps"],
        )
        checkpoint_score += d_cp

        if _check_collision((x, y), centerline, left_bound, right_bound):
            collision = True
            break

    steer_changes = np.sum(np.abs(np.diff(steer_list))) if len(steer_list) > 1 else 0
    n_steps = step + 1
    avg_speed = total_dist / (n_steps * dt) if n_steps > 0 else 0.0
    survival_time = float(n_steps * dt)
    return collision, total_dist, avg_speed, min_radar, steer_changes, survival_time, checkpoint_score


def _simulate_episode_with_path(track, params, max_steps):
    """
    用给定策略仿真一轮，并记录轨迹点，用于可视化。
    返回 (collision, total_dist, avg_speed, min_radar, steer_changes, survival_time, checkpoint_score, path_xy)。
    path_xy: list of (x, y)，长度为步数+1（含起点）。
    """
    centerline, left_bound, right_bound, gates = unpack_track(track)
    cp_cfg = config.CHECKPOINT
    n_gates = len(gates) if gates is not None else 0
    cp_state = CheckpointState.empty(n_gates)
    checkpoint_score = 0.0
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
    step = -1
    for step in range(max_steps):
        radar = get_radar_distances((x, y), theta, left_bound, right_bound)
        min_radar = min(min_radar, np.min(radar))
        steer, target_speed = _params_to_steer_speed(params, radar, prev_steer)
        steer_list.append(steer)
        prev_steer = steer
        px, py = x, y
        v = v + config.CAR["accel"] * (target_speed - v)
        v = np.clip(v, config.CAR["min_speed"], config.CAR["max_speed"])
        omega = v * np.tan(np.clip(steer, -max_steer, max_steer)) / wheelbase
        theta += omega * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        total_dist += v * dt
        d_cp, cp_state = score_checkpoint_crossing(
            (px, py),
            (x, y),
            gates,
            cp_state,
            cp_cfg["pass_bonus"],
            cp_cfg["wrong_penalty"],
            cp_cfg["cooldown_steps"],
        )
        checkpoint_score += d_cp
        path_xy.append((x, y))
        if _check_collision((x, y), centerline, left_bound, right_bound):
            collision = True
            break
    steer_changes = np.sum(np.abs(np.diff(steer_list))) if len(steer_list) > 1 else 0
    n_steps = step + 1
    avg_speed = total_dist / (n_steps * dt) if n_steps > 0 else 0.0
    survival_time = float(n_steps * dt)
    return collision, total_dist, avg_speed, min_radar, steer_changes, survival_time, checkpoint_score, path_xy


def _fitness(collision, total_dist, avg_speed, min_radar, steer_changes, survival_time, checkpoint_score):
    """
    适应度：撞墙为 0；否则为加权和（距离、速度、存活时间、离墙）减去转向抖动惩罚。
    权重来自 config.PSO["fitness_weights"]。
    """
    if collision:
        return 0.0
    w = config.PSO.get(
        "fitness_weights",
        {
            "total_dist": 0.5,
            "avg_speed": 4.0,
            "min_radar": 0.005,
            "survival_time": 8.0,
            "steer_penalty": 0.01,
            "checkpoint": 1.0,
        },
    )
    # 无有效步数时 min_radar 可能仍为 inf，用雷达最大量程替代避免数值问题
    if not np.isfinite(min_radar):
        min_radar = float(config.RADAR.get("max_distance", 800.0))

    f = (
        total_dist * w["total_dist"]
        + avg_speed * w["avg_speed"]
        + min_radar * w["min_radar"]
        + survival_time * w["survival_time"]
        + float(checkpoint_score) * w.get("checkpoint", 1.0)
    )
    f -= steer_changes * w["steer_penalty"]
    return max(0.0, f)


def _draw_pso_trajectories_live(ax, track, X, gbest_idx, vis_steps, max_trajectories=6, colors=None):
    """
    在给定 axes 上绘制赛道与部分粒子的轨迹（仅画 max_trajectories 条，每条用 vis_steps 步，加快刷新）。
    gbest_idx: 当前全局最优粒子索引，其轨迹用粗线高亮。
    """
    import matplotlib.pyplot as _plt
    centerline, left_bound, right_bound, gates = unpack_track(track)
    n_particles = len(X)
    # 只画最优 + 前 max_trajectories-1 个粒子，避免每轮仿真过多
    indices = [gbest_idx] + [i for i in range(n_particles) if i != gbest_idx][: max_trajectories - 1]
    if colors is None:
        colors = _plt.cm.tab10(np.linspace(0, 1, max(10, len(indices))))
    # 赛道
    ax.plot(centerline[:, 0], centerline[:, 1], "k-", lw=1, alpha=0.8)
    ax.plot(left_bound[:, 0], left_bound[:, 1], "gray", lw=0.8, alpha=0.6)
    ax.plot(right_bound[:, 0], right_bound[:, 1], "gray", lw=0.8, alpha=0.6)
    if gates is not None and len(gates) > 0:
        for gi in range(len(gates)):
            ax.plot(
                [gates[gi, 0, 0], gates[gi, 1, 0]],
                [gates[gi, 0, 1], gates[gi, 1, 1]],
                color="magenta",
                lw=1.0,
                alpha=0.65,
            )
    # 仅对选中粒子仿真并画轨迹（用 vis_steps 步，加快）
    for k, i in enumerate(indices):
        _, _, _, _, _, _, _, path_xy = _simulate_episode_with_path(track, X[i], vis_steps)
        path_xy = np.array(path_xy)
        c = colors[k % len(colors)]
        lw = 2.5 if i == gbest_idx else 0.8
        alpha = 1.0 if i == gbest_idx else 0.5
        ax.plot(path_xy[:, 0], path_xy[:, 1], "-", color=c, lw=lw, alpha=alpha)
    ax.set_aspect("equal")
    ax.autoscale(True)


def run_pso(track, max_steps=None, n_particles=None, n_iters=None, visualize=None):
    """
    在给定赛道上运行 PSO，搜索最优驾驶策略参数。

    Parameters
    ----------
    track : tuple of (centerline, left_bound, right_bound[, checkpoint_gates])
    max_steps : int, optional
    n_particles : int, optional
    n_iters : int, optional
    visualize : bool, optional
        True 时每轮迭代实时绘制所有粒子轨迹（弹窗更新）；默认从 config.PSO["visualize_training"] 读取。

    Returns
    -------
    best_params : np.ndarray, shape (6,)
    fitness_history : list of float
    """
    cfg = config.PSO
    max_steps = max_steps or cfg["max_steps_per_episode"]
    n_particles = n_particles or cfg["n_particles"]
    n_iters = n_iters or cfg["n_iters"]
    do_visualize = visualize if visualize is not None else cfg.get("visualize_training", False)
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
    gbest_idx = 0
    w = cfg["w"]
    c1 = cfg["c1"]
    c2 = cfg["c2"]
    fitness_history = []

    # 实时可视化：先创建并显示空窗口，再在循环中更新（避免“加载好久不显示”）
    fig, ax = None, None
    vis_steps = max_steps
    max_traj = 6
    update_every = 1
    if do_visualize:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))
            ax.set_title("PSO Training - Particle Trajectories")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            centerline, left_bound, right_bound, gates = unpack_track(track)
            ax.plot(centerline[:, 0], centerline[:, 1], "k-", lw=1, alpha=0.8)
            ax.plot(left_bound[:, 0], left_bound[:, 1], "gray", lw=0.8, alpha=0.6)
            ax.plot(right_bound[:, 0], right_bound[:, 1], "gray", lw=0.8, alpha=0.6)
            if gates is not None and len(gates) > 0:
                for gi in range(len(gates)):
                    ax.plot(
                        [gates[gi, 0, 0], gates[gi, 1, 0]],
                        [gates[gi, 0, 1], gates[gi, 1, 1]],
                        color="magenta",
                        lw=1.0,
                        alpha=0.65,
                    )
            ax.set_aspect("equal")
            ax.autoscale(True)
            fig.show()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.02)
            vis_steps = min(max_steps, cfg.get("visualize_steps", 80))
            max_traj = cfg.get("visualize_max_trajectories", 6)
            update_every = max(1, int(cfg.get("visualize_update_every", 1)))
        except ImportError:
            do_visualize = False

    for it in range(n_iters):
        for i in range(n_particles):
            collision, total_dist, avg_speed, min_radar, steer_changes, survival_time, checkpoint_score = _simulate_episode(
                track, X[i], max_steps, rng
            )
            f = _fitness(
                collision, total_dist, avg_speed, min_radar, steer_changes, survival_time, checkpoint_score
            )
            if f > pbest_f[i]:
                pbest_f[i] = f
                pbest[i] = X[i].copy()
            if f > gbest_f:
                gbest_f = f
                gbest = X[i].copy()
                gbest_idx = i
        fitness_history.append(gbest_f)
        print(f"  PSO 迭代 {it + 1}/{n_iters}  当前最优适应度: {gbest_f:.2f}", flush=True)

        # 按配置间隔更新图，只画少量轨迹、较少步数，加快刷新
        if do_visualize and fig is not None and ax is not None and (it % update_every == 0):
            ax.clear()
            ax.set_title(f"Iter {it + 1}/{n_iters}  Best Fitness: {gbest_f:.2f}")
            _draw_pso_trajectories_live(ax, track, X, gbest_idx, vis_steps, max_traj)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.03)

        # 更新速度与位置
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        for d in range(dim):
            X[:, d] = np.clip(X[:, d], low[d], high[d])

    if do_visualize and fig is not None:
        try:
            plt.ioff()
            plt.show()
        except Exception:
            pass

    return gbest, fitness_history


def plot_pso_results(track, best_params, fitness_history, save_path=None, show=True):
    """
    PSO 结果可视化：适应度曲线 + 最优策略在赛道上的轨迹。

    Parameters
    ----------
    track : tuple of (centerline, left_bound, right_bound[, checkpoint_gates])
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
    centerline, left_bound, right_bound, gates = unpack_track(track)
    max_steps = config.PSO.get("max_steps_per_episode", 200)
    _, _, _, _, _, _, _, path_xy = _simulate_episode_with_path(track, best_params, max_steps)
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
    if gates is not None and len(gates) > 0:
        for gi in range(len(gates)):
            ax2.plot(
                [gates[gi, 0, 0], gates[gi, 1, 0]],
                [gates[gi, 0, 1], gates[gi, 1, 1]],
                "m-",
                lw=1.2,
                alpha=0.75,
            )
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
