# -*- coding: utf-8 -*-
"""
Pygame 仿真可视化：实时绘制赛道、车辆、16 条雷达射线，HUD 显示速度/转向/雷达距离。
支持 PSO 专家模式与神经网络自动驾驶模式；控制器接口 (state) -> (steer, speed) 便于扩展。
"""
import numpy as np
import pygame

import config
from sensor.radar import get_radar_distances


def _world_to_screen(x, y, sim_cfg):
    """世界坐标 (x, y)，y 向上为正 → 屏幕坐标，y 向下为正。"""
    sx = sim_cfg["offset_x"] + x * sim_cfg["scale"]
    sy = sim_cfg["offset_y"] - y * sim_cfg["scale"]
    return int(sx), int(sy)


def _draw_track(screen, track, sim_cfg):
    """绘制赛道：填充左右边界之间的区域，再画中心线。"""
    centerline, left_bound, right_bound = track
    # 赛道填充（左边界 + 右边界逆序 形成闭合多边形）
    pts = [_world_to_screen(p[0], p[1], sim_cfg) for p in left_bound]
    pts += [_world_to_screen(p[0], p[1], sim_cfg) for p in right_bound[::-1]]
    if len(pts) >= 3:
        pygame.draw.polygon(screen, sim_cfg["track_left_color"], pts)
    # 中心线
    for i in range(len(centerline)):
        p1 = _world_to_screen(centerline[i][0], centerline[i][1], sim_cfg)
        p2 = _world_to_screen(centerline[(i + 1) % len(centerline)][0], centerline[(i + 1) % len(centerline)][1], sim_cfg)
        pygame.draw.line(screen, sim_cfg["track_center_color"], p1, p2, 2)


def _draw_car(screen, x, y, heading, sim_cfg):
    """绘制车辆：三角形，顶点朝车头方向。"""
    L = 12
    angles = [heading, heading + 2.5, heading - 2.5]
    pts = [
        (x + L * np.cos(angles[0]), y + L * np.sin(angles[0])),
        (x + L * 0.6 * np.cos(angles[1]), y + L * 0.6 * np.sin(angles[1])),
        (x + L * 0.6 * np.cos(angles[2]), y + L * 0.6 * np.sin(angles[2])),
    ]
    pts_screen = [_world_to_screen(p[0], p[1], sim_cfg) for p in pts]
    pygame.draw.polygon(screen, sim_cfg["car_color"], pts_screen)
    pygame.draw.polygon(screen, (255, 255, 255), pts_screen, 1)


def _draw_radar(screen, x, y, heading, radar_distances, sim_cfg):
    """绘制 16 条雷达射线，命中处用不同颜色。"""
    n_rays = len(radar_distances)
    angle_step = 2 * np.pi / n_rays
    for i in range(n_rays):
        angle = heading + i * angle_step
        d = radar_distances[i]
        end_x = x + d * np.cos(angle)
        end_y = y + d * np.sin(angle)
        start_s = _world_to_screen(x, y, sim_cfg)
        end_s = _world_to_screen(end_x, end_y, sim_cfg)
        color = sim_cfg["radar_hit_color"] if d < config.RADAR["max_distance"] * 0.99 else sim_cfg["radar_ray_color"]
        pygame.draw.line(screen, color, start_s, end_s, 1)


def _draw_hud(screen, speed, steer, radar_distances, mode_str, sim_cfg, font):
    """绘制 HUD：速度、转向、模式；可选 16 个雷达距离。"""
    y_pos = 20
    text = font.render(f"模式: {mode_str}  |  速度: {speed:.2f}  |  转向: {steer:.3f}", True, sim_cfg["hud_text_color"])
    screen.blit(text, (20, y_pos))
    if sim_cfg.get("show_radar_values", True) and radar_distances is not None:
        radar_str = " ".join([f"{d:.0f}" for d in radar_distances])
        text2 = font.render(f"雷达: {radar_str}", True, sim_cfg["hud_text_color"])
        screen.blit(text2, (20, y_pos + 22))


def run_simulation(track, mode="pso", pso_params=None, model=None):
    """
    运行 Pygame 仿真循环。

    Parameters
    ----------
    track : tuple of (centerline, left_bound, right_bound)
    mode : str
        "pso" 或 "nn"
    pso_params : np.ndarray, shape (6,), optional
        PSO 模式下的策略参数。
    model : 已训练模型, optional
        NN 模式下的 sklearn 或 PyTorch 模型。
    """
    pygame.init()
    sim_cfg = config.SIM
    car_cfg = config.CAR
    centerline, left_bound, right_bound = track

    screen = pygame.display.set_mode((sim_cfg["window_width"], sim_cfg["window_height"]))
    pygame.display.set_caption("端到端自动驾驶仿真 - PSO/神经网络")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # 初始状态：中心线起点，朝向沿赛道
    start = centerline[0]
    diff = centerline[1] - centerline[0]
    start_heading = np.arctan2(diff[1], diff[0])
    x, y = float(start[0]), float(start[1])
    theta = start_heading
    v = car_cfg["min_speed"]
    dt = car_cfg["dt"]
    max_steer = car_cfg["max_steer"]
    wheelbase = car_cfg["wheelbase"]
    prev_steer = 0.0

    # 控制器：扩展接口 (state) -> (steer, speed)，state = (radar_16, speed)
    if mode == "pso" and pso_params is not None:
        from pso.optimizer import _params_to_steer_speed
        def controller(radar, speed):
            return _params_to_steer_speed(pso_params, radar, prev_steer)
    elif mode == "nn" and model is not None:
        from model.network import predict
        def controller(radar, speed):
            X = np.array([list(radar) + [speed]], dtype=np.float64)
            out = predict(model, X)
            return float(out[0, 0]), float(out[0, 1])
    else:
        def controller(radar, speed):
            return 0.0, car_cfg["min_speed"]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        radar = get_radar_distances((x, y), theta, left_bound, right_bound)
        steer, target_speed = controller(radar, v)
        prev_steer = steer

        v = v + car_cfg["accel"] * (target_speed - v)
        v = np.clip(v, car_cfg["min_speed"], car_cfg["max_speed"])
        omega = v * np.tan(np.clip(steer, -max_steer, max_steer)) / wheelbase
        theta += omega * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt

        screen.fill(sim_cfg["bg_color"])
        _draw_track(screen, track, sim_cfg)
        _draw_radar(screen, x, y, theta, radar, sim_cfg)
        _draw_car(screen, x, y, theta, sim_cfg)
        _draw_hud(screen, v, steer, radar if sim_cfg.get("show_radar_values") else None, mode.upper(), sim_cfg, font)
        pygame.display.flip()
        clock.tick(sim_cfg["fps"])

    pygame.quit()
