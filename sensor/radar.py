# -*- coding: utf-8 -*-
"""
16 方向雷达测距：以车为中心 0°～360° 均匀 16 条射线，
计算每条射线到赛道边界的最近交点距离。
"""
import numpy as np

import config


def _ray_segment_intersection(origin, direction, seg_start, seg_end, max_dist):
    """
    射线与线段求交。射线: origin + t * direction, t >= 0，direction 已单位化。
    线段: seg_start + u * (seg_end - seg_start), u in [0,1]。
    返回交点距离 t（单位化后即长度），若无交点返回 None。
    """
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-10)
    A = np.asarray(seg_start, dtype=float)
    B = np.asarray(seg_end, dtype=float)
    # O + t*d = A + u*(B-A)  =>  [ (B-A)_x  -d_x ] [u]   [O_x - A_x]
    #                             [ (B-A)_y  -d_y ] [t] = [O_y - A_y]
    M = np.array([[B[0] - A[0], -d[0]], [B[1] - A[1], -d[1]]])
    denom = np.linalg.det(M)
    if abs(denom) < 1e-10:
        return None
    rhs = o - A
    u = (rhs[0] * (-d[1]) - rhs[1] * (-d[0])) / denom
    t = ((B[0] - A[0]) * rhs[1] - (B[1] - A[1]) * rhs[0]) / denom
    if 0 <= u <= 1 and t >= 0 and t <= max_dist:
        return float(t)
    return None


def _bound_to_segments(bound):
    """将边界点数组转为首尾相连的线段列表 [(p0,p1), (p1,p2), ...]。"""
    segs = []
    n = len(bound)
    for i in range(n):
        segs.append((bound[i], bound[(i + 1) % n]))
    return segs


def get_radar_distances(car_pos, car_heading, left_bound, right_bound):
    """
    计算 16 个方向的雷达测距值。

    Parameters
    ----------
    car_pos : array-like, shape (2,)
        车辆位置 (x, y)。
    car_heading : float
        车辆朝向角（弧度），0 为 x 轴正方向，逆时针为正。
    left_bound : np.ndarray, shape (n, 2)
        赛道左边界点。
    right_bound : np.ndarray, shape (n, 2)
        赛道右边界点。

    Returns
    -------
    np.ndarray, shape (16,)
        16 个方向上的距离，单位与坐标一致；无交点时为 max_distance。
    """
    n_rays = config.RADAR["n_rays"]
    max_dist = config.RADAR["max_distance"]

    car_pos = np.asarray(car_pos, dtype=float)
    left_segs = _bound_to_segments(left_bound)
    right_segs = _bound_to_segments(right_bound)
    all_segs = left_segs + right_segs

    distances = np.zeros(n_rays)
    angle_step = 2 * np.pi / n_rays

    for i in range(n_rays):
        # 射线在世界坐标系下的方向：先按雷达角 0~360°，再叠加车头朝向
        ray_angle = car_heading + i * angle_step
        direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
        min_d = max_dist
        for seg in all_segs:
            d = _ray_segment_intersection(
                car_pos, direction, seg[0], seg[1], max_dist
            )
            if d is not None and d < min_d:
                min_d = d
        distances[i] = min_d

    return distances.astype(np.float64)
