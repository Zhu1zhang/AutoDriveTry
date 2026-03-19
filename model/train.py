# -*- coding: utf-8 -*-
"""
神经网络训练：从专家数据加载，训练 17→32→64→2 回归模型，保存模型与训练曲线。
"""
import os
import numpy as np

import config
from .network import create_model, predict


def _load_data(data_path):
    """从目录加载 expert_X.npy、expert_Y.npy 或 expert_data.csv。"""
    data_path = os.path.abspath(data_path)
    npy_x = os.path.join(data_path, "expert_X.npy")
    npy_y = os.path.join(data_path, "expert_Y.npy")
    if os.path.isfile(npy_x) and os.path.isfile(npy_y):
        X = np.load(npy_x)
        Y = np.load(npy_y)
        return X, Y
    csv_path = os.path.join(data_path, "expert_data.csv")
    if os.path.isfile(csv_path):
        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = list(reader)
        data = np.array([[float(x) for x in row] for row in rows])
        # 前 17 列为输入，后 2 列为输出
        X = data[:, :17]
        Y = data[:, 17:19]
        return X, Y
    raise FileNotFoundError(f"未在 {data_path} 中找到 expert_X.npy/expert_Y.npy 或 expert_data.csv")


def train(data_path, model_save_path, cfg=None):
    """
    加载专家数据，训练模型，保存到 model_save_path，并可选保存训练曲线图。

    Parameters
    ----------
    data_path : str
        专家数据目录（含 expert_X.npy / expert_Y.npy 或 expert_data.csv）。
    model_save_path : str
        模型保存路径（.pkl 或 .pt）。
    cfg : dict, optional
        覆盖 config.MODEL 的配置。

    Returns
    -------
    model : 训练好的模型
    """
    cfg = cfg or config.MODEL
    backend = cfg.get("backend", "sklearn")
    max_iter = cfg.get("max_iter", 500)

    X, Y = _load_data(data_path)
    if len(X) < 10:
        raise ValueError("专家数据样本过少，请先运行 PSO 并采集更多数据。")

    # 简单划分 train（无单独 val，仅用 train loss 画图）
    n = len(X)
    indices = np.random.RandomState(cfg.get("random_state", 42)).permutation(n)
    X, Y = X[indices], Y[indices]

    model = create_model(backend=backend)

    if backend == "sklearn":
        # 用 partial_fit 模拟 epoch 以记录 loss 曲线
        from sklearn.neural_network import MLPRegressor
        batch_size = min(64, len(X))
        train_losses = []
        for epoch in range(max_iter):
            model.partial_fit(X, Y)
            pred = model.predict(X)
            mse = np.mean((pred - Y) ** 2)
            train_losses.append(mse)
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}/{max_iter}  Train MSE: {mse:.6f}")

        os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
        import joblib
        joblib.dump(model, model_save_path)
        print(f"  模型已保存: {model_save_path}")

        # 训练曲线
        _plot_curve(train_losses, model_save_path)
        return model

    if backend == "pytorch":
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        # 设备：优先使用 GPU
        dev_cfg = cfg.get("device", "auto")
        if dev_cfg == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(dev_cfg)
        print(f"  训练设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
        model = create_model(backend="pytorch")
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.get("learning_rate_init", 0.001))
        Xt = torch.from_numpy(X).float()
        Yt = torch.from_numpy(Y).float()
        loader = DataLoader(TensorDataset(Xt, Yt), batch_size=64, shuffle=True)
        train_losses = []
        for epoch in range(max_iter):
            model.train()
            total_loss = 0.0
            for xi, yi in loader:
                xi, yi = xi.to(device), yi.to(device)
                opt.zero_grad()
                out = model(xi)
                loss = nn.functional.mse_loss(out, yi)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            mse = total_loss / len(loader)
            train_losses.append(mse)
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}/{max_iter}  Train MSE: {mse:.6f}")
        os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
        # 保存到 CPU 以便推理时任意设备加载
        torch.save(model.cpu().state_dict(), model_save_path)
        print(f"  模型已保存: {model_save_path}")
        _plot_curve(train_losses, model_save_path)
        return model

    return model


def _plot_curve(train_losses, model_save_path):
    """将 train loss 绘制为 train_curve.png，与模型同目录。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    dirname = os.path.dirname(model_save_path)
    out_path = os.path.join(dirname, "train_curve.png") if dirname else "train_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, color="C0", label="Train MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Training Loss (MSE)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  训练曲线已保存: {out_path}")
