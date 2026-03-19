# -*- coding: utf-8 -*-
"""
随机赛道生成：样条扰动法。
基础形状为随机椭圆，施加径向扰动后经 B 样条平滑，得到曲率连续、无尖角的中心线，
再沿法向偏移生成左右边界。
"""
import numpy as np
from scipy.interpolate import splprep, splev

import config


def _sample_ellipse(a, b, cx, cy, rotation, n_points):
    """在椭圆上均匀取 n_points 个点（按角度均匀）。"""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    # 旋转
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    pts = np.dot(np.column_stack([x, y]), R.T) + np.array([cx, cy])
    return pts


def _perturb_points(pts, scale, seed=None):
    """对每个点施加径向随机扰动，避免自交。"""
    if seed is not None:
        np.random.seed(seed)
    n = len(pts)
    # 中心
    center = pts.mean(axis=0)
    # 径向单位向量（指向外）
    vec = pts - center
    dist = np.linalg.norm(vec, axis=1, keepdims=True)
    dist[dist < 1e-6] = 1e-6
    radial = vec / dist
    # 径向扰动
    noise = np.random.randn(n, 1) * scale
    pts_new = pts + radial * noise
    return pts_new


def _smooth_centerline(pts, smoothness, n_out):
    """B 样条平滑闭合曲线，重采样为 n_out 个点。"""
    # 闭合：首尾相接
    pts_closed = np.vstack([pts, pts[0:1]])
    # splprep 要求 (2, n) 或 (n, 2)；我们用 (n, 2)
    tck, u = splprep(pts_closed.T, s=smoothness, per=1, k=3)
    u_new = np.linspace(0, 1, n_out, endpoint=False)
    x_new, y_new = splev(u_new, tck)
    centerline = np.column_stack([x_new, y_new])
    return centerline


def _compute_normals(centerline):
    """计算中心线每点的单位法向量（指向左侧）。"""
    n = len(centerline)
    d = np.diff(centerline, axis=0)
    d = np.vstack([d, centerline[0] - centerline[-1]])  # 闭合
    # 切向 (归一化)
    tang = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    # 2D 法向：切向 (dx, dy) -> 左侧法向 (-dy, dx)
    normals = np.column_stack([-tang[:, 1], tang[:, 0]])
    return normals


def generate_track(seed=None):
    """
    生成一条随机平滑赛道。

    Parameters
    ----------
    seed : int, optional
        随机种子，便于复现。

    Returns
    -------
    centerline : np.ndarray, shape (n_centerline_points, 2)
        中心线点坐标。
    left_bound : np.ndarray, shape (n_centerline_points, 2)
        左边界点坐标。
    right_bound : np.ndarray, shape (n_centerline_points, 2)
        右边界点坐标。
    """
    cfg = config.TRACK
    if seed is not None:
        np.random.seed(seed)

    # 随机椭圆参数
    a = np.random.uniform(cfg["ellipse_a_min"], cfg["ellipse_a_max"])
    b = np.random.uniform(cfg["ellipse_b_min"], cfg["ellipse_b_max"])
    cx = cfg["center_x"]
    cy = cfg["center_y"]
    rotation = np.random.uniform(0, 2 * np.pi)

    # 椭圆采样
    pts = _sample_ellipse(
        a, b, cx, cy, rotation, cfg["n_sample_points"]
    )

    # 径向扰动
    pts = _perturb_points(pts, cfg["perturbation_scale"], seed=None)

    # B 样条平滑并重采样
    centerline = _smooth_centerline(
        pts,
        cfg["spline_smoothness"],
        cfg["n_centerline_points"],
    )

    # 法向与左右边界
    normals = _compute_normals(centerline)
    half = cfg["half_width"]
    left_bound = centerline + half * normals
    right_bound = centerline - half * normals

    return centerline, left_bound, right_bound
