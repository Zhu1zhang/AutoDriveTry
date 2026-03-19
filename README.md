# 端到端自动驾驶训练与仿真平台

纯 Python、无 ROS、轻量级的端到端自动驾驶训练与仿真平台：随机生成平滑赛道，用粒子群算法 (PSO) 搜索最优驾驶策略，采集专家数据训练神经网络，实现 16 方向雷达测距的端到端自动驾驶，并在 Pygame 中可视化。

## 动图演示

<!-- 可将运行仿真时的录屏保存为 docs/demo.gif 后取消下一行注释 -->
<!-- ![仿真演示](docs/demo.gif) -->

（运行 `python main.py` 后即可看到 PSO 专家驾驶与神经网络自动驾驶的实时仿真。）

## 功能亮点

- **随机赛道生成**：样条扰动法，基于随机椭圆 + B 样条平滑，曲率连续、无尖角，自动生成中心线与左右边界。
- **16 向雷达**：以车为中心 0°～360° 均匀 16 条射线，计算到赛道边界的距离，输出 16 维距离向量。
- **PSO 策略优化**：粒子群算法搜索驾驶策略参数，适应度考虑离边界距离、速度、碰撞与轨迹平滑。
- **专家数据采集**：用 PSO 最优策略跑赛道，每帧记录 17 维输入（16 雷达 + 速度）与 2 维输出（转向、目标速度），保存为 CSV/NumPy。
- **神经网络训练**：17→32→64→2 全连接网络（ReLU），回归转向与速度，支持 scikit-learn 或 PyTorch，训练曲线可视化。
- **Pygame 仿真**：实时绘制赛道、车辆、雷达射线与 HUD，支持 PSO 专家模式与神经网络自动驾驶模式。

## 安装

```bash
git clone <your-repo-url>
cd F1-project
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

**使用阿里云镜像（国内推荐）**：

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

项目根目录下的 `pip.ini` 已配置阿里云镜像，若将 `PIP_CONFIG_FILE` 指向该文件，或将其复制到用户 pip 配置目录，则后续 `pip install` 会默认使用阿里云源。

依赖：`numpy`、`pygame`、`torch`、`scikit-learn`、`scipy`、`joblib`、`matplotlib`（可选，用于训练曲线）。

**使用 GPU 训练**：默认已启用 PyTorch，有 NVIDIA GPU 时会自动使用 CUDA。若需指定设备，在 `config.py` 的 `MODEL` 中设置 `"device": "cuda"` 或 `"cpu"`；安装 CUDA 版 PyTorch 见 [PyTorch 官网](https://pytorch.org/get-started/locally/)。

**环境自检**：安装完成后可运行自检脚本，验证依赖与各模块是否正常：

```bash
python check_env.py
```

## 运行

**一键完整流程（推荐）**：生成赛道 → PSO 优化 → 采集专家数据 → 训练网络 → 启动仿真（先 PSO 演示，再 NN 模式）

```bash
python main.py
```

**可选参数**：

- `--seed 42`：随机种子（赛道与 PSO）
- `--skip-pso`：跳过 PSO，使用默认参数采集（需确保已有或接受少量样本）
- `--skip-train`：跳过训练，仅运行仿真（需已有 `data/models/mlp.pkl`）
- `--sim-only`：仅运行仿真，加载已有模型（会按当前 seed 重新生成赛道）
- `--no-sim`：不启动 Pygame，只执行 PSO + 采集 + 训练
- `--data-dir data/expert_data`：专家数据目录
- `--model-path data/models/mlp.pkl`：模型保存/加载路径

## 项目结构

```
F1-project/
├── main.py                 # 主入口，一键运行
├── config.py               # 全局配置（赛道/雷达/PSO/网络/仿真）
├── requirements.txt
├── README.md
├── track/
│   └── generator.py        # 样条扰动法随机赛道生成
├── sensor/
│   └── radar.py            # 16 方向雷达测距
├── pso/
│   └── optimizer.py        # PSO 驾驶策略搜索
├── expert/
│   └── collector.py        # 专家数据采集（CSV/NumPy）
├── model/
│   ├── network.py          # 网络结构 17→32→64→2
│   └── train.py            # 训练与保存、训练曲线
├── sim/
│   └── pygame_sim.py       # Pygame 仿真与双模式控制
└── data/                   # 运行时生成（可 .gitignore）
    ├── expert_data/
    └── models/
```

## 算法原理简述

- **赛道**：椭圆离散点 → 径向随机扰动 → B 样条闭合平滑 → 沿法向偏移得左右边界。
- **雷达**：射线与线段求交，取最近交点距离，无交点返回最大探测距离。
- **PSO**：粒子为 6 维策略参数（左/右雷达权重、速度系数、转向偏置等），在赛道上仿真得到适应度（距离、速度、碰撞、平滑），迭代更新位置与速度。
- **网络**：输入 17 维（16 雷达 + 速度），隐藏 32/64 ReLU，输出 2 维（转向角、目标速度），MSE 回归；可选 PyTorch 或 sklearn MLPRegressor。

## 开源协议

MIT License.
