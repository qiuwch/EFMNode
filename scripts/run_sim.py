#!/usr/bin/env python3
"""
EFMNode 仿真/真机 模式运行脚本

支持:
- 一键切换真机/仿真模式
- 接入 GalaxeaManipSim 仿真环境
- Pseudo random policy 测试
- 可视化渲染

用法:
    # 仿真模式 + 随机策略
    python scripts/run_sim.py --mode sim --random-policy
    
    # 仿真模式 + 可视化
    python scripts/run_sim.py --mode sim --random-policy --render
    
    # 真机模式 (需要 ROS2)
    python scripts/run_sim.py --mode real
    
    # 指定环境
    python scripts/run_sim.py --mode sim --env R1ProBlocksStackEasy-v0 --steps 500
"""

import argparse
import sys
import os
import time
import numpy as np
import toml
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from galaxea_fm.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


def setup_logger():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )


def load_config(model_path: str = None):
    """加载配置"""
    default_config_path = Path(__file__).parent.parent / "config.toml"
    
    config = toml.load(default_config_path)
    
    # 覆盖模型路径（如果提供）
    if model_path:
        config.setdefault("model", {})
        config["model"]["ckpt_dir"] = model_path
    
    return config


def run_simulation(
    mode: str = "sim",
    env_name: str = "R1ProBlocksStackEasy-v0",
    use_random_policy: bool = True,
    random_seed: int = 42,
    num_steps: int = 500,
    render: bool = False,
    headless: bool = None,
    control_freq: float = 15.0,
):
    """
    运行仿真或真机
    
    Args:
        mode: 运行模式 ("sim" 或 "real")
        env_name: Gym 环境名称 (仅仿真模式)
        use_random_policy: 是否使用随机策略
        random_seed: 随机种子
        num_steps: 运行步数
        render: 是否渲染
        headless: 无头模式（默认根据 render 推断）
        control_freq: 控制频率 Hz
    """
    setup_logger()
    logger.info("=" * 60)
    if mode == "sim":
        logger.info("EFMNode - Simulation Mode 🤖")
    else:
        logger.info("EFMNode - Real Robot Mode 🦾")
    logger.info("=" * 60)
    
    # 加载配置
    config = load_config()
    
    # 设置运行模式
    config.setdefault("robot", {})
    config["robot"]["run_mode"] = mode
    
    # 添加仿真配置
    config.setdefault("simulation", {})
    config["simulation"]["enabled"] = (mode == "sim")
    config["simulation"]["env_name"] = env_name
    config["simulation"]["use_random_policy"] = use_random_policy
    config["simulation"]["random_seed"] = random_seed
    config["simulation"]["headless"] = not render if headless is None else headless
    
    logger.info(f"Mode: {mode}")
    if mode == "sim":
        logger.info(f"Environment: {env_name}")
        logger.info(f"Random Policy: {use_random_policy}")
        logger.info(f"Random Seed: {random_seed}")
        logger.info(f"Render: {render}")
    logger.info(f"Control Frequency: {control_freq} Hz")
    logger.info(f"Steps: {num_steps}")
    
    # 根据模式导入不同的桥接
    if mode == "sim":
        from core.communication.sim_bridge import SimBridge, PseudoRandomPolicy
        
        try:
            bridge = SimBridge(
                config=config,
                cfg={},
                env_name=env_name,
                use_random_policy=use_random_policy,
                random_seed=random_seed,
                headless=config["simulation"]["headless"],
            )
        except ImportError as e:
            logger.error(f"Failed to initialize SimBridge: {e}")
            logger.error("Make sure GalaxeaManipSim is installed:")
            logger.error("  pip install -e ../GalaxeaManipSim")
            sys.exit(1)
    else:
        # 真机模式 - 使用 Scheduler 自动处理
        from scheduler.scheduler import Scheduler
        logger.info("Initializing ROS2 bridge for real robot...")
        scheduler = Scheduler(config)
        bridge = scheduler.ros2_bridge
    
    # 运行循环
    logger.info(f"Starting for {num_steps} steps...")
    step_count = 0
    start_time = time.time()
    
    try:
        while step_count < num_steps and bridge.is_running():
            if mode == "sim":
                if use_random_policy:
                    # 使用内置随机策略
                    bridge.step_with_random_policy()
                else:
                    # 获取观测（可用于外部策略）
                    obs_time, obs = bridge.gather_obs()
                    
                    if obs is not None:
                        # 这里可以接入你自己的策略
                        # action = your_policy(obs)
                        # bridge.publish_action(action)
                        logger.warning("No policy specified, using random actions")
                        bridge.step_with_random_policy()
                    else:
                        logger.warning("No observation received")
            else:
                # 真机模式 - 运行 scheduler
                obs_time, obs = scheduler.ros2_bridge.gather_obs()
                if obs is not None:
                    actions = scheduler.inference(obs)
                    if actions is not None and scheduler.cnt >= 2:
                        scheduler.step(actions['action'], obs_time)
                scheduler.cnt += 1
            
            step_count += 1
            
            # 进度报告
            if step_count % 50 == 0:
                elapsed = time.time() - start_time
                fps = step_count / elapsed if elapsed > 0 else 0
                logger.info(f"Step {step_count}/{num_steps} ({fps:.1f} FPS)")
            
            # 控制频率
            if control_freq > 0:
                time.sleep(1.0 / control_freq)
        
        # 完成报告
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        mode_name = "Simulation" if mode == "sim" else "Real Robot"
        logger.info(f"{mode_name} completed!")
        logger.info(f"Total steps: {step_count}")
        logger.info(f"Elapsed time: {elapsed:.2f}s")
        logger.info(f"Average FPS: {step_count / elapsed:.1f}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        bridge.destroy()


def main():
    parser = argparse.ArgumentParser(description="EFMNode - Simulation/Real Robot Mode")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sim", "real"],
        default=None,
        help="Run mode: 'sim' (simulation) or 'real' (real robot). Overrides config.toml"
    )
    
    parser.add_argument(
        "--env", "--env-name",
        type=str,
        default="R1ProBlocksStackEasy-v0",
        help="Gym environment name (default: R1ProBlocksStackEasy-v0, sim mode only)"
    )
    
    parser.add_argument(
        "--random-policy",
        action="store_true",
        help="Use pseudo random policy for testing (sim mode)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of steps to run (default: 500)"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable visualization (default: headless, sim mode only)"
    )
    
    parser.add_argument(
        "--freq",
        type=float,
        default=15.0,
        help="Control frequency in Hz (default: 15.0)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (for VLA inference)"
    )
    
    args = parser.parse_args()
    
    # 确定运行模式：命令行 > config.toml
    config = load_config(args.model_path)
    mode = args.mode if args.mode else config.get('robot', {}).get('run_mode', 'sim')
    
    run_simulation(
        mode=mode,
        env_name=args.env,
        use_random_policy=args.random_policy,
        random_seed=args.seed,
        num_steps=args.steps,
        render=args.render,
        control_freq=args.freq,
    )


if __name__ == "__main__":
    main()
