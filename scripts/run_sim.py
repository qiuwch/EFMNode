#!/usr/bin/env python3
"""
EFMNode 仿真模式运行脚本

支持:
- 接入 GalaxeaManipSim 仿真环境
- Pseudo random policy 测试
- 可视化渲染

用法:
    # 运行随机策略测试
    python scripts/run_sim.py --env R1ProBlocksStackEasy-v0 --random-policy
    
    # 带可视化
    python scripts/run_sim.py --env R1ProBlocksStackEasy-v0 --random-policy --render
    
    # 指定步数
    python scripts/run_sim.py --env R1ProBlocksStackEasy-v0 --random-policy --steps 500
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
    env_name: str = "R1ProBlocksStackEasy-v0",
    use_random_policy: bool = True,
    random_seed: int = 42,
    num_steps: int = 500,
    render: bool = False,
    headless: bool = None,
    control_freq: float = 15.0,
):
    """
    运行仿真
    
    Args:
        env_name: Gym 环境名称
        use_random_policy: 是否使用随机策略
        random_seed: 随机种子
        num_steps: 运行步数
        render: 是否渲染
        headless: 无头模式（默认根据 render 推断）
        control_freq: 控制频率 Hz
    """
    setup_logger()
    logger.info("=" * 60)
    logger.info("EFMNode Simulation Mode")
    logger.info("=" * 60)
    
    # 加载配置
    config = load_config()
    
    # 添加仿真配置
    config.setdefault("simulation", {})
    config["simulation"]["enabled"] = True
    config["simulation"]["env_name"] = env_name
    config["simulation"]["use_random_policy"] = use_random_policy
    config["simulation"]["random_seed"] = random_seed
    config["simulation"]["headless"] = not render if headless is None else headless
    
    logger.info(f"Environment: {env_name}")
    logger.info(f"Random Policy: {use_random_policy}")
    logger.info(f"Random Seed: {random_seed}")
    logger.info(f"Render: {render}")
    logger.info(f"Control Frequency: {control_freq} Hz")
    
    # 导入 SimBridge
    from core.communication.sim_bridge import SimBridge, PseudoRandomPolicy
    
    # 创建 SimBridge
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
    
    # 运行循环
    logger.info(f"Starting simulation for {num_steps} steps...")
    step_count = 0
    start_time = time.time()
    
    try:
        while step_count < num_steps and bridge.is_running():
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
        logger.info(f"Simulation completed!")
        logger.info(f"Total steps: {step_count}")
        logger.info(f"Elapsed time: {elapsed:.2f}s")
        logger.info(f"Average FPS: {step_count / elapsed:.1f}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        bridge.destroy()


def main():
    parser = argparse.ArgumentParser(description="EFMNode Simulation Mode")
    
    parser.add_argument(
        "--env", "--env-name",
        type=str,
        default="R1ProBlocksStackEasy-v0",
        help="Gym environment name (default: R1ProBlocksStackEasy-v0)"
    )
    
    parser.add_argument(
        "--random-policy",
        action="store_true",
        help="Use pseudo random policy for testing"
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
        help="Enable visualization (default: headless)"
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
        help="Path to model checkpoint (for future VLA integration)"
    )
    
    args = parser.parse_args()
    
    run_simulation(
        env_name=args.env,
        use_random_policy=args.random_policy,
        random_seed=args.seed,
        num_steps=args.steps,
        render=args.render,
        control_freq=args.freq,
    )


if __name__ == "__main__":
    main()
