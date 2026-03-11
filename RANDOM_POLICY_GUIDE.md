# EFMNode 随机策略数据收集指南

## 概述

本文档说明如何在仿真环境中使用随机策略收集演示数据，用于：
- 测试数据管道
- 生成 baseline 数据
- 验证仿真环境配置

---

## 快速开始

### 1. 安装依赖

```bash
cd ~/efmnode

# 确保 GalaxeaManipSim 已安装
pip install -e ../GalaxeaManipSim

# 安装额外依赖
pip install h5py loguru gymnasium
```

### 2. 收集随机演示

```bash
# 基础用法 - 收集 50 个演示
python scripts/collect_random_demos.py \
    --env-name R1ProBlocksStackEasy-v0 \
    --num-demos 50

# 指定输出目录
python scripts/collect_random_demos.py \
    --env-name R1ProBlocksStackEasy-v0 \
    --num-demos 100 \
    --dataset-dir ./my_datasets
```

### 3. 使用高级随机策略

```bash
# 随机游走模式（推荐 - 更平滑）
python scripts/random_policy.py \
    --env-name R1ProBlocksStackEasy-v0 \
    --mode random-walk

# 带噪声的 PD 控制
python scripts/random_policy.py \
    --env-name R1ProBlocksStackEasy-v0 \
    --mode noisy-pd \
    --noise-scale 0.3

# 完全随机
python scripts/random_policy.py \
    --env-name R1ProBlocksStackEasy-v0 \
    --mode random
```

---

## 支持的环境

| 环境名称 | 机器人 | 任务 |
|---------|--------|------|
| `R1ProBlocksStackEasy-v0` | R1 Pro | 堆方块（简单） |
| `R1ProBlocksStackHard-v0` | R1 Pro | 堆方块（困难） |
| `R1ProDualBottlesPickEasy-v0` | R1 Pro | 抓双瓶子 |
| `R1DualBottlesPickEasy-v0` | R1 | 抓双瓶子 |
| `R1LiteBlocksStackEasy-v0` | R1 Lite | 堆方块 |
| `R1MugHangingEasy-v0` | R1 | 挂杯子 |
| `R1DualShoesPlace-v0` | R1 | 放鞋子 |

---

## 随机策略模式

### `random` - 完全随机
- 每帧独立采样随机动作
- 动作范围：初始位置 ±0.15 rad
- 适合快速测试

### `random-walk` - 随机游走 ⭐推荐
- 动作平滑连续
- 步长限制：0.05 rad/步
- 更适合物理仿真

### `noisy-pd` - 噪声 PD 控制
- PD 控制器跟踪随机目标
- 添加 Ornstein-Uhlenbeck 噪声
- 目标每 50 步更新一次

### `sinusoidal` - 正弦波探索
- 使用正弦波生成周期性动作
- 适合探索周期性运动

---

## 数据格式

### HDF5 结构

```
demo_0000.h5
├── attrs:
│   ├── env_name: "R1ProBlocksStackEasy-v0"
│   ├── policy: "random"
│   ├── num_steps: 200
│   └── timestamp: "2026-03-11T..."
├── image_head_rgb: (200, 224, 224, 3) uint8
├── image_left_wrist_rgb: (200, 224, 224, 3) uint8
├── image_right_wrist_rgb: (200, 224, 224, 3) uint8
├── obs_qpos: (200, 7) float32
├── obs_left_arm_joint_position: (200, 7) float32
├── obs_right_arm_joint_position: (200, 7) float32
├── obs_left_arm_gripper_position: (200, 1) float32
├── obs_right_arm_gripper_position: (200, 1) float32
├── obs_left_arm_ee_pose: (200, 7) float32
├── obs_right_arm_ee_pose: (200, 7) float32
├── actions: (200, 16) float32
├── rewards: (200,) float32
├── terminated: (200,) bool
└── truncated: (200,) bool
```

### 元数据 (meta_info.json)

```json
[
  {
    "episode_id": 1,
    "num_steps": 200,
    "num_tries": 1,
    "success_rate": "100.0%",
    "timestamp": "2026-03-11T15:30:00"
  }
]
```

---

## 可视化数据

### 查看 HDF5 内容

```python
import h5py
import matplotlib.pyplot as plt

with h5py.File('datasets/R1ProBlocksStackEasy-v0/random_policy/demo_0000.h5', 'r') as f:
    print("Keys:", list(f.keys()))
    print("Actions shape:", f['actions'].shape)
    
    # 绘制动作轨迹
    actions = f['actions'][:]
    plt.figure(figsize=(10, 6))
    plt.plot(actions[:, :7], label='Left arm')
    plt.plot(actions[:, 7:14], label='Right arm')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('Joint position')
    plt.title('Random Policy Actions')
    plt.show()
```

### 回放演示

```bash
# 使用 GalaxeaManipSim 回放
python -m galaxea_sim.scripts.replay_demos \
    --env-name R1ProBlocksStackEasy-v0 \
    --target-controller-type bimanual_joint_position \
    --num-demos 1
```

---

## 转换为 LeRobot 格式

```bash
# 转换为 LeRobot 数据集
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot \
    --task R1ProBlocksStackEasy-v0 \
    --tag random_policy \
    --robot r1_pro
```

---

## 高级用法

### 自定义随机策略

```python
from scripts.random_policy import RandomPolicy
import gymnasium as gym
import galaxea_sim.envs

env = gym.make("R1ProBlocksStackEasy-v0", headless=False)
policy = RandomPolicy(env, mode="random-walk", noise_scale=0.2)

obs, _ = env.reset()
for i in range(200):
    action = policy.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, _ = env.reset()
        policy.reset()

env.close()
```

### 批量收集

```bash
# 并行收集（多个终端）
for i in {1..5}; do
    python scripts/collect_random_demos.py \
        --env-name R1ProBlocksStackEasy-v0 \
        --num-demos 20 \
        --dataset-dir ./datasets_run_$i &
done
wait
```

---

## 故障排除

### 问题：`ModuleNotFoundError: No module named 'galaxea_sim'`

```bash
# 确保已安装 GalaxeaManipSim
cd ~/GalaxeaManipSim
pip install -e .
```

### 问题：`Asset not found`

```bash
# 下载仿真资产
cd ~/GalaxeaManipSim
gdown https://drive.google.com/file/d/1ZvtCv1H4FLrse_ElUWzsVDt8xRK4CyaC/
unzip robotwin_models.zip
mv robotwin_models galaxea_sim/assets/
```

### 问题：动作维度不匹配

检查机器人配置：
- R1/R1 Lite: 6 DOF 手臂 → 动作维度 14
- R1 Pro: 7 DOF 手臂 → 动作维度 16

---

## 下一步

1. **验证数据质量**: 检查随机动作的分布和范围
2. **转换为训练格式**: 使用 LeRobot 转换脚本
3. **训练 Baseline**: 用随机数据训练简单的行为克隆模型
4. **对比实验**: 与专家演示对比性能差距

---

## 参考

- [GalaxeaManipSim README](../GalaxeaManipSim/README.md)
- [EFMNode README](README.md)
- [LeRobot 文档](https://github.com/huggingface/lerobot)
