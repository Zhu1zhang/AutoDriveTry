# -*- coding: utf-8 -*-
"""
神经网络部分快速测试（无 Pygame）：
  1) 检查能否加载模型；
  2) 对随机/专家样本做 predict，检查输出形状与数值范围。

用法（在 AutoDriveTry 目录下）:
  python test_nn.py
  python test_nn.py --model-path data/models/mlp.pkl
  python test_nn.py --with-expert   # 若有 expert_X.npy，顺带算与专家标签的 MAE
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model.network import predict


def main():
    parser = argparse.ArgumentParser(description="测试神经网络加载与 predict")
    parser.add_argument("--model-path", type=str, default="data/models/mlp.pkl")
    parser.add_argument("--data-dir", type=str, default="data/expert_data")
    parser.add_argument(
        "--with-expert",
        action="store_true",
        help="若存在 expert_X/Y.npy，计算预测与专家输出的 MAE",
    )
    args = parser.parse_args()
    path = args.model_path
    if not os.path.isfile(path):
        print(f"[失败] 未找到模型: {path}")
        print("  请先运行: python main.py（完整流程）或 python main.py --skip-pso（需已有专家数据）")
        return 1

    backend = config.MODEL.get("backend", "sklearn")
    print(f"后端: {backend}  |  模型: {path}")

    if backend == "sklearn":
        import joblib
        model = joblib.load(path)
    else:
        import torch
        from model.network import create_model
        model = create_model(backend="pytorch")
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

    # 虚拟输入：雷达约 0~max_distance，速度在 min~max_speed 之间
    rng = np.random.default_rng(0)
    radar_max = config.RADAR["max_distance"]
    n = 5
    X = rng.uniform(0, radar_max, size=(n, 16))
    spd = rng.uniform(
        config.CAR["min_speed"],
        config.CAR["max_speed"],
        size=(n, 1),
    )
    X = np.hstack([X, spd])

    out = predict(model, X, backend=backend)
    assert out.shape == (n, 2), out.shape
    print(f"[OK] predict 输出形状 (n, 2) = {out.shape}")
    print("  样例前 2 条 (steer, target_speed):")
    for i in range(min(2, n)):
        print(f"    {out[i]}")

    if args.with_expert:
        dx = os.path.join(args.data_dir, "expert_X.npy")
        dy = os.path.join(args.data_dir, "expert_Y.npy")
        if os.path.isfile(dx) and os.path.isfile(dy):
            Xe = np.load(dx)
            Ye = np.load(dy)
            pred = predict(model, Xe, backend=backend)
            mae = np.mean(np.abs(pred - Ye))
            print(f"[OK] 专家集 MAE (全样本): {mae:.6f}")
        else:
            print(f"[跳过] 未找到 {dx} 或 {dy}")

    print("\n完整可视化测试请运行: python main.py --sim-only")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
