# -*- coding: utf-8 -*-
"""
端到端自动驾驶训练与仿真平台 - 主入口。
一键运行：生成赛道 → PSO 优化 → 采集专家数据 → 训练神经网络 → Pygame 仿真（PSO 演示 + 神经网络模式）。
支持命令行参数选择仅运行部分步骤或仅仿真。
"""
import os
import sys
import argparse
import json

import numpy as np

# 将项目根目录加入路径，便于直接 python main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from track import generate_track, unpack_track
from pso import run_pso, plot_pso_results
from expert import collect_expert_data
from model.network import create_model, predict
from model.train import train as run_train
from sim import run_simulation


def parse_args():
    parser = argparse.ArgumentParser(description="端到端自动驾驶训练与仿真平台")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于赛道生成与 PSO")
    parser.add_argument("--skip-pso", action="store_true", help="跳过 PSO，使用已有专家数据")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，仅运行仿真（需已有模型）")
    parser.add_argument("--sim-only", action="store_true", help="仅运行仿真，加载已有模型与赛道数据（需提前生成）")
    parser.add_argument("--no-sim", action="store_true", help="不启动 Pygame，仅执行 PSO + 采集 + 训练")
    parser.add_argument("--pso-visualize", action="store_true", help="PSO 训练时实时显示每个粒子轨迹（弹窗逐轮更新）")
    parser.add_argument("--no-pso-visualize", action="store_true", help="关闭 PSO 实时轨迹（覆盖 config 中的 visualize_training，适合后台训练）")
    parser.add_argument("--data-dir", type=str, default="data/expert_data", help="专家数据目录")
    parser.add_argument("--model-path", type=str, default="data/models/mlp.pkl", help="模型保存/加载路径")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    model_path = args.model_path
    os.makedirs(os.path.dirname(model_path) or "data/models", exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # ---------- 1. 生成赛道 ----------
    print("【1/6】生成随机赛道...")
    track = generate_track(seed=args.seed)
    centerline, left_bound, right_bound, gates = unpack_track(track)
    n_cp = len(gates) if gates is not None else 0
    print(f"  中心线点数: {len(centerline)}, 半宽: {config.TRACK['half_width']}, 检查点: {n_cp}")

    if args.sim_only:
        # 仅仿真模式：加载已有模型（赛道按当前 seed 重新生成）
        print("【仅仿真】使用当前 seed 生成赛道，加载已有模型...")
        if not os.path.isfile(model_path):
            print("  错误：未找到模型文件，请先运行完整流程或去掉 --sim-only")
            return
        backend = config.MODEL.get("backend", "sklearn")
        if backend == "sklearn":
            import joblib
            model = joblib.load(model_path)
        else:
            import torch
            from model.network import create_model
            model = create_model(backend="pytorch")
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        run_simulation(track, mode="nn", model=model)
        return

    # ---------- 2. PSO 优化 ----------
    if not args.skip_pso:
        print("【2/6】PSO 优化驾驶策略...")
        pso_viz = (args.pso_visualize or config.PSO.get("visualize_training", False)) and not args.no_pso_visualize
        best_params, fitness_history = run_pso(
            track,
            n_iters=config.PSO["n_iters"],
            n_particles=config.PSO["n_particles"],
            visualize=pso_viz,
        )
        print(f"  最优适应度: {fitness_history[-1]:.2f}")
        # PSO 可视化：适应度曲线 + 最优轨迹
        pso_fig_path = os.path.join(os.path.dirname(model_path) or "data/models", "pso_result.png")
        plot_pso_results(
            track, best_params, fitness_history,
            save_path=pso_fig_path,
            show=config.PSO.get("visualize_show", False),
        )
        # 持久化本次训练/结果图所用赛道与 seed，便于用相同 generate_track(seed) 复现
        models_dir = os.path.dirname(pso_fig_path) or "data/models"
        meta_path = os.path.join(models_dir, "pso_track_meta.json")
        npz_path = os.path.join(models_dir, "training_track.npz")
        meta = {
            "seed": args.seed,
            "generator": "track.generate_track (polar smooth perturbation)",
            "track_config": {
                "n_theta": config.TRACK["n_theta"],
                "base_radius": config.TRACK["base_radius"],
                "half_width": config.TRACK["half_width"],
                "min_radius_ratio": config.TRACK["min_radius_ratio"],
                "smooth_window": config.TRACK["smooth_window"],
                "n_checkpoints": config.TRACK.get("n_checkpoints", 0),
            },
            "checkpoint_scoring": {
                "pass_bonus": config.CHECKPOINT["pass_bonus"],
                "wrong_penalty": config.CHECKPOINT["wrong_penalty"],
                "cooldown_steps": config.CHECKPOINT["cooldown_steps"],
            },
            "note": "Re-run with the same --seed to get the same track as pso_result.png",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        cl, lb, rb, g = unpack_track(track)
        save_kw = dict(centerline=cl, left_bound=lb, right_bound=rb, seed=args.seed)
        if g is not None and getattr(g, "size", 0) > 0:
            save_kw["checkpoint_gates"] = g
        np.savez_compressed(npz_path, **save_kw)
        print(f"  赛道元数据: {meta_path}")
        print(f"  赛道几何存档: {npz_path}")
    else:
        print("【2/6】跳过 PSO，将使用已有专家数据训练（需确保已有采集数据）")
        # 若无已有参数，用默认参数做一次采集以便仅训练
        import numpy as np
        bounds = np.array(config.PSO["param_bounds"])
        best_params = (bounds[:, 0] + bounds[:, 1]) / 2

    # ---------- 3. 采集专家数据 ----------
    print("【3/6】采集专家数据...")
    X, Y = collect_expert_data(track, best_params, data_dir)
    print(f"  采集样本数: {len(X)}")
    if len(X) < 10:
        print("  警告：样本过少，训练可能不稳定。建议取消 --skip-pso 重新运行。")

    # ---------- 4. 训练神经网络 ----------
    if not args.skip_train:
        print("【4/6】训练神经网络...")
        try:
            run_train(data_dir, model_path)
        except FileNotFoundError as e:
            print(f"  错误: {e}")
            return
    else:
        print("【4/6】跳过训练")

    # ---------- 5/6. 仿真 ----------
    if not args.no_sim:
        print("【5/6】启动仿真 - PSO 专家模式...")
        run_simulation(track, mode="pso", pso_params=best_params)
        if os.path.isfile(model_path):
            print("【6/6】启动仿真 - 神经网络模式...")
            backend = config.MODEL.get("backend", "sklearn")
            if backend == "sklearn":
                import joblib
                model = joblib.load(model_path)
            else:
                import torch
                from model.network import create_model
                model = create_model(backend="pytorch")
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            run_simulation(track, mode="nn", model=model)
        else:
            print("【6/6】未找到模型文件，跳过神经网络仿真。")
    else:
        print("【5/6】【6/6】已跳过仿真 (--no-sim)")


if __name__ == "__main__":
    main()
