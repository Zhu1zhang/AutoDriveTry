# -*- coding: utf-8 -*-
"""
赛道生成：极坐标平滑扰动法。天生闭合、无交叉、曲率连续，仅依赖 numpy。
"""
import numpy as np
import config


def _moving_average_wrap(arr, window):
    """对闭合序列做移动平均（首尾循环），窗口为奇数。"""
    n = len(arr)
    w = int(window) // 2  # 半窗
    kernel = np.ones(2 * w + 1) / (2 * w + 1)
    # 首尾各补 w 个点以循环
    padded = np.concatenate([arr[-w:], arr, arr[:w]])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _centerline_normals_closed(x, y):
    """
    中心差分法求闭合中心线每点单位法向（指向左侧）。
    首点：dx = x[1]-x[-1], dy = y[1]-y[-1]；尾点：dx = x[0]-x[-2], dy = y[0]-y[-2]；中间点：dx = x[i+1]-x[i-1], dy = y[i+1]-y[i-1]。
    """
    n = len(x)
    dx = np.empty(n)
    dy = np.empty(n)
    dx[0] = x[1] - x[-1]
    dy[0] = y[1] - y[-1]
    dx[-1] = x[0] - x[-2]
    dy[-1] = y[0] - y[-2]
    dx[1:-1] = x[2:] - x[:-2]
    dy[1:-1] = y[2:] - y[:-2]
    norm = np.sqrt(dx * dx + dy * dy) + 1e-10
    nx = -dy / norm
    ny = dx / norm
    return np.column_stack([nx, ny])


def generate_closed_track(n_theta=None, base_radius=None, half_width=None, seed=None):
    """
    极坐标平滑扰动法生成闭合赛道。天生无交叉、首尾衔接、曲率连续、边界等距。

    参数
    ------
    n_theta : int
        极角采样点数（150-200）。
    base_radius : float
        基础半径（像素）。
    half_width : float
        赛道半宽（像素）。
    seed : int, optional
        随机种子；当前扰动为确定性公式，seed 保留接口兼容。

    返回
    ------
    centerline : np.ndarray, shape (N, 2)
    left_bound : np.ndarray, shape (N, 2)
    right_bound : np.ndarray, shape (N, 2)
    """
    cfg = config.TRACK
    n_theta = n_theta or cfg.get("n_theta", 180)
    base_radius = base_radius if base_radius is not None else cfg["base_radius"]
    half_width = half_width if half_width is not None else cfg["half_width"]
    min_ratio = cfg.get("min_radius_ratio", 0.6)
    terms = cfg.get("perturbation_terms", [(30, 2), (20, 3), (15, 5)])
    smooth_window = cfg.get("smooth_window", 5)
    if smooth_window % 2 == 0:
        smooth_window += 1

    # 若提供 seed，用其随机生成一组扰动项，使每次赛道形状不同
    if seed is not None:
        np.random.seed(seed)
        n_terms = np.random.randint(2, 6)
        terms = [(np.random.uniform(15, 40), np.random.randint(2, 7)) for _ in range(n_terms)]

    # 1. 极角均匀采样，360° 闭合（endpoint=True 使首尾同为 2π，严格闭合）
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=True)

    # 2. 平滑半径：基础半径 + 2-5 个低频 sin/cos 叠加，无高频噪声
    r = np.full_like(theta, base_radius, dtype=np.float64)
    for amp, freq in terms:
        r += amp * np.cos(freq * theta)
    r = np.maximum(r, base_radius * min_ratio)

    # 3. 转直角坐标，天生首尾闭合、无交叉
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # 4. 二次平滑：移动平均，曲率连续、无尖角
    x = _moving_average_wrap(x, smooth_window)
    y = _moving_average_wrap(y, smooth_window)

    centerline = np.column_stack([x, y])
    # 强制首尾同点，保证 100% 闭合（消除浮点误差）
    centerline[-1] = centerline[0]

    # 5. 中心差分法求法向，首尾单独处理保证闭合连续
    normals = _centerline_normals_closed(centerline[:, 0], centerline[:, 1])

    # 6. 左右边界等距偏移，全程无错位
    left_bound = centerline + half_width * normals
    right_bound = centerline - half_width * normals
    # 强制左右边界首尾一致，保证闭合
    left_bound[-1] = left_bound[0]
    right_bound[-1] = right_bound[0]

    return (
        centerline.astype(np.float64),
        left_bound.astype(np.float64),
        right_bound.astype(np.float64),
    )


def generate_track(seed=None):
    """对外接口：生成闭合赛道，与雷达、PSO、仿真兼容。返回 centerline, left_bound, right_bound。"""
    return generate_closed_track(seed=seed)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    c, l, r = generate_closed_track(seed=42)
    plt.figure(figsize=(7, 7))
    plt.plot(c[:, 0], c[:, 1], "k--", lw=1.5, label="Centerline")
    plt.plot(l[:, 0], l[:, 1], "b-", lw=1, label="Left bound")
    plt.plot(r[:, 0], r[:, 1], "b-", lw=1, label="Right bound")
    plt.axis("equal")
    plt.legend()
    plt.title("Closed Track (Polar Smooth Perturbation)")
    plt.tight_layout()
    plt.show()
