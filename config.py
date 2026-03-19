# -*- coding: utf-8 -*-
"""
全局配置文件：赛道、雷达、PSO、神经网络、仿真参数集中管理。
便于调参与扩展，各模块从此处读取配置。
"""

# ========== 赛道生成（极坐标平滑扰动法，天生闭合无交叉） ==========
TRACK = {
    "n_theta": 180,             # 极角采样点数（150-200），360° 闭合
    "base_radius": 150.0,       # 基础半径（像素）
    "half_width": 35,           # 赛道半宽（像素），左右边界与中心线等距
    "min_radius_ratio": 0.6,   # 半径最小值 = base_radius * min_radius_ratio，避免过度收缩
    # 低频扰动系数：仅 2-5 个低频 sin/cos 叠加，无高频噪声
    # 格式：列表 [(幅度, 角频率), ...]，如 (30, 2) 表示 30*cos(2*theta)
    "perturbation_terms": [(30, 2), (20, 3), (15, 5)],
    "smooth_window": 5,         # 中心线二次平滑的移动平均窗口大小（奇数）
}

# ========== 16 向雷达 ==========
RADAR = {
    "n_rays": 16,              # 射线数量
    "max_distance": 800.0,     # 最大探测距离（像素），无交点时返回此值
}

# ========== 车辆运动学（用于 PSO 与仿真） ==========
CAR = {
    "wheelbase": 8,            # 轴距（像素，用于转向几何）
    "max_steer": 0.6,          # 最大转向角（弧度）
    "max_speed": 6.0,          # 最大速度
    "min_speed": 0.5,          # 最小速度
    "accel": 0.15,             # 加速度
    "dt": 0.1,                 # 仿真时间步长（秒）
}

# ========== PSO 驾驶策略优化 ==========
PSO = {
    "n_particles": 15,         # 粒子数（减小可加快首次运行）
    "n_iters": 20,             # 迭代次数（减小可加快首次运行）
    "max_steps_per_episode": 200,  # 每条粒子每轮仿真最大步数（减小可加快）
    "w": 0.7,                  # 惯性权重
    "c1": 1.5,                 # 个体学习因子
    "c2": 1.5,                 # 社会学习因子
    # 策略参数边界 [min, max]，粒子维数 = 6：左前/右前权重、速度系数、转向偏置等
    "param_bounds": [
        (-2.0, 2.0),   # k_steer_left
        (-2.0, 2.0),   # k_steer_right
        (0.2, 1.0),    # k_speed
        (-0.5, 0.5),   # bias_steer
        (0.0, 1.0),    # min_dist_weight（最近距离权重）
        (-0.3, 0.3),   # steer_smooth（平滑项）
    ],
    "visualize_show": False,  # PSO 完成后是否弹窗显示可视化（True 会阻塞到关窗）
}

# ========== 专家数据采集 ==========
EXPERT = {
    "max_frames": 10000,       # 单次采集最大帧数
    "max_laps": 3,             # 最大圈数（可选，按圈截断）
    "output_csv": True,        # 是否输出 CSV
    "output_npy": True,        # 是否输出 NumPy
}

# ========== 神经网络 ==========
MODEL = {
    "backend": "pytorch",     # "sklearn"（仅 CPU）或 "pytorch"（支持 GPU）
    "device": "auto",          # "auto"（有 GPU 则用）/ "cuda" / "cpu"
    "input_dim": 17,           # 16 雷达 + 1 速度
    "hidden_sizes": [32, 64],  # 隐藏层神经元数
    "output_dim": 2,           # 转向、目标速度
    "max_iter": 500,           # 训练迭代次数
    "learning_rate_init": 0.001,
    "random_state": 42,
}

# ========== Pygame 仿真 ==========
SIM = {
    "window_width": 900,
    "window_height": 700,
    "scale": 1.0,              # 世界坐标到屏幕的缩放（可调）
    "offset_x": 450,            # 世界原点在屏幕上的 x
    "offset_y": 350,            # 世界原点在屏幕上的 y
    "show_radar_values": True,  # 是否在 HUD 显示 16 个雷达距离
    "fps": 60,
    "bg_color": (20, 24, 32),
    "track_center_color": (80, 90, 100),
    "track_left_color": (40, 50, 60),
    "track_right_color": (40, 50, 60),
    "car_color": (60, 180, 120),
    "radar_ray_color": (100, 200, 255),
    "radar_hit_color": (255, 180, 80),
    "hud_text_color": (220, 220, 220),
}
