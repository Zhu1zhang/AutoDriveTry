# -*- coding: utf-8 -*-
r"""
环境与模块自检脚本：检查依赖、导入、各模块最小功能。
运行: python check_env.py  或  venv\Scripts\python check_env.py
"""
import sys
import os

# 确保项目根在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def ok(msg):
    print(f"  [OK] {msg}")
def fail(msg):
    print(f"  [FAIL] {msg}")
    return False

def main():
    print("========== 1. Python 版本 ==========")
    print(f"  {sys.version}")
    if sys.version_info < (3, 8):
        fail("需要 Python 3.8+")
        return
    ok("Python 版本符合要求")

    print("\n========== 2. 依赖包 ==========")
    deps = [
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("joblib", "joblib"),
        ("pygame", "pygame 或 pygame-ce"),
        ("matplotlib", "matplotlib"),
    ]
    for mod, name in deps:
        try:
            __import__(mod)
            ok(f"{name}")
        except ImportError as e:
            fail(f"{name}: {e}")
            return

    print("\n========== 3. 项目模块导入 ==========")
    try:
        import config
        ok("config")
        from track import generate_track
        ok("track.generate_track")
        from sensor.radar import get_radar_distances
        ok("sensor.get_radar_distances")
        from pso.optimizer import run_pso
        ok("pso.run_pso")
        from expert.collector import collect_expert_data
        ok("expert.collect_expert_data")
        from model.network import create_model, predict
        ok("model.network")
        from model.train import train
        ok("model.train")
        from sim.pygame_sim import run_simulation
        ok("sim.run_simulation")
    except Exception as e:
        fail(f"模块导入: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n========== 4. 赛道生成 ==========")
    try:
        track = generate_track(seed=42)
        c, l, r = track
        assert len(c) > 0 and c.shape[1] == 2
        assert len(l) == len(c) and len(r) == len(c)
        ok(f"中心线 {len(c)} 点, 左右边界正常")
    except Exception as e:
        fail(f"赛道生成: {e}")
        return

    print("\n========== 5. 雷达测距 ==========")
    try:
        import numpy as np
        d = get_radar_distances((0.0, 0.0), 0.0, l, r)
        assert d.shape == (16,), d.shape
        assert np.all(d >= 0)
        ok(f"16 维距离, 范围 [0, max]")
    except Exception as e:
        fail(f"雷达: {e}")
        return

    print("\n========== 6. PSO 短跑（2 迭代, 4 粒子, 30 步） ==========")
    try:
        best, hist = run_pso(track, n_iters=2, n_particles=4, max_steps=30)
        assert best.shape == (6,), best.shape
        assert len(hist) == 2
        ok(f"最优适应度 {hist[-1]:.2f}")
    except Exception as e:
        fail(f"PSO: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n========== 7. 专家数据采集 ==========")
    try:
        out_dir = "data/expert_data_check"
        os.makedirs(out_dir, exist_ok=True)
        X, Y = collect_expert_data(track, best, out_dir, max_frames=100)
        assert X.shape[1] == 17 and Y.shape[1] == 2
        ok(f"采集 {len(X)} 条, X(17), Y(2)")
    except Exception as e:
        fail(f"专家采集: {e}")
        return

    print("\n========== 8. 神经网络训练（10 epoch） ==========")
    try:
        import config as cfg_mod
        model_path = "data/models/check_model.pkl"
        os.makedirs("data/models", exist_ok=True)
        train_cfg = {**cfg_mod.MODEL, "max_iter": 10, "random_state": 42}
        train(out_dir, model_path, cfg=train_cfg)
        assert os.path.isfile(model_path)
        ok("模型已保存")
    except Exception as e:
        fail(f"训练: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n========== 9. 模型预测 ==========")
    try:
        import joblib
        from model.network import predict
        model = joblib.load(model_path)
        out = predict(model, X[:3])
        assert out.shape == (3, 2)
        ok("predict 输出 (3, 2)")
    except Exception as e:
        fail(f"预测: {e}")
        return

    print("\n========== 自检通过 ==========")
    print("可运行: python main.py  （完整流程）")
    print("或:     python main.py --no-sim  （不启动仿真窗口）")

if __name__ == "__main__":
    main()
