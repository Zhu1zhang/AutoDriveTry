# -*- coding: utf-8 -*-
"""
神经网络结构：输入 17（16 雷达 + 1 速度）→ 隐藏 32/64 (ReLU) → 输出 2（转向、速度）。
支持 sklearn MLPRegressor 或扩展 PyTorch。
"""
import numpy as np

import config


def create_model(backend=None, hidden_sizes=None):
    """
    创建未训练的模型。用于训练脚本或扩展接口。

    Parameters
    ----------
    backend : str, optional
        "sklearn" 或 "pytorch"，默认从 config 读取。
    hidden_sizes : list of int, optional
        隐藏层神经元数，默认 config.MODEL["hidden_sizes"]。

    Returns
    -------
    model : 可 fit/predict 的对象（sklearn）或 nn.Module（pytorch）
    """
    backend = backend or config.MODEL["backend"]
    hidden_sizes = hidden_sizes or config.MODEL["hidden_sizes"]
    in_dim = config.MODEL["input_dim"]
    out_dim = config.MODEL["output_dim"]

    if backend == "sklearn":
        from sklearn.neural_network import MLPRegressor
        # hidden_layer_sizes=(32, 64)
        model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_sizes),
            activation="relu",
            solver="adam",
            max_iter=1,  # 由 train 中循环控制
            learning_rate_init=config.MODEL.get("learning_rate_init", 0.001),
            random_state=config.MODEL.get("random_state", 42),
            warm_start=True,
        )
        return model
    if backend == "pytorch":
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("请安装 PyTorch: pip install torch")
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        model = nn.Sequential(*layers)
        return model
    raise ValueError("backend 应为 'sklearn' 或 'pytorch'")


def predict(model, X, backend=None):
    """
    单次或批量预测。X shape (n, 17)，返回 (n, 2)。

    Parameters
    ----------
    model : 已训练模型
    X : np.ndarray, shape (n, 17)
    backend : str, optional

    Returns
    -------
    np.ndarray, shape (n, 2)
    """
    backend = backend or config.MODEL["backend"]
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if backend == "sklearn":
        return model.predict(X).astype(np.float64)
    if backend == "pytorch":
        import torch
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            x = torch.from_numpy(X).float().to(device)
            y = model(x)
            return y.cpu().numpy().astype(np.float64)
    raise ValueError("backend 应为 'sklearn' 或 'pytorch'")
