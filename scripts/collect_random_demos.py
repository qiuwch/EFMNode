#!/usr/bin/env python3
"""
在仿真环境中使用随机策略收集演示数据

用法:
    python scripts/collect_random_demos.py --env-name R1ProBlocksStackEasy-v0 --num-demos 50
"""

import argparse
import numpy as np
import gymnasium as gym
import h5py
import json
import time
from pathlib import Path
from datetime import datetime
from loguru import logger

# 导入仿真环境
import galaxea_sim.envs
from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv


class RandomPolicy:
    """随机策略生成器"""
    
    def __init__(self, env: BimanualManipulationEnv):
        self.env = env
        self.action_space = env.action_space
        
        # 获取动作维度信息
        self.left_arm_dim = len(env.left_arm_joint_indices)
        self.right_arm_dim = len(env.right_arm_joint_indices)
        self.action_dim = self.left_arm_dim + self.right_arm_dim + 2  # +2 for grippers
        
        logger.info(f"Action space dim: {self.action_dim}")
        logger.info(f"Left arm DOF: {self.left_arm_dim}, Right arm DOF: {self.right_arm_dim}")
        
        # 从初始位置开始的小范围随机
        self.init_qpos = env.init_qpos.copy()
        self.left_arm_init = self.init_qpos[env.left_arm_joint_indices]
        self.right_arm_init = self.init_qpos[env.right_arm_joint_indices]
        self.gripper_init = np.array([0.5, 0.5])  # 半开夹爪
        
    def get_action(self, obs=None):
        """生成随机动作"""
        # 在初始位置附近生成小范围随机动作
        left_arm = self.left_arm_init + np.random.uniform(-0.1, 0.1, self.left_arm_dim)
        right_arm = self.right_arm_init + np.random.uniform(-0.1, 0.1, self.right_arm_dim)
        left_gripper = np.clip(0.5 + np.random.uniform(-0.2, 0.2), 0, 1)
        right_gripper = np.clip(0.5 + np.random.uniform(-0.2, 0.2), 0, 1)
        
        action = np.concatenate([
            left_arm,
            [left_gripper],
            right_arm,
            [right_gripper]
        ]).astype(np.float32)
        
        return action
    
    def reset(self):
        """重置策略状态"""
        pass


class DemoCollector:
    """演示数据收集器"""
    
    def __init__(self, env_name: str, dataset_dir: str = "datasets"):
        self.env_name = env_name
        self.dataset_dir = Path(dataset_dir) / env_name / "random_policy"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建环境
        self.env = gym.make(env_name, headless=False, obs_mode="image")
        self.unwrapped_env = self.env.unwrapped
        assert isinstance(self.unwrapped_env, BimanualManipulationEnv)
        
        # 创建随机策略
        self.policy = RandomPolicy(self.unwrapped_env)
        
        # 统计数据
        self.num_collected = 0
        self.num_tries = 0
        
    def collect_observation(self, obs):
        """将观测转换为可保存的格式"""
        upper_obs = obs.get('upper_body_observations', {})
        
        saved_obs = {}
        
        # 保存图像
        for key in ['head_rgb', 'left_wrist_rgb', 'right_wrist_rgb']:
            if key in upper_obs:
                img = upper_obs[key]
                if isinstance(img, np.ndarray):
                    saved_obs[f'image_{key}'] = img.transpose(1, 2, 0)  # CHW -> HWC
                else:
                    saved_obs[f'image_{key}'] = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # 保存状态
        saved_obs['qpos'] = upper_obs.get('left_arm_joint_position', np.zeros(7))
        saved_obs['left_arm_joint_position'] = upper_obs.get('left_arm_joint_position', np.zeros(7))
        saved_obs['right_arm_joint_position'] = upper_obs.get('right_arm_joint_position', np.zeros(7))
        saved_obs['left_arm_gripper_position'] = upper_obs.get('left_arm_gripper_position', np.zeros(1))
        saved_obs['right_arm_gripper_position'] = upper_obs.get('right_arm_gripper_position', np.zeros(1))
        saved_obs['left_arm_ee_pose'] = upper_obs.get('left_arm_ee_pose', np.zeros(7))
        saved_obs['right_arm_ee_pose'] = upper_obs.get('right_arm_ee_pose', np.zeros(7))
        
        return saved_obs
    
    def collect_episode(self, max_steps: int = 200):
        """收集单个 episode"""
        traj = []
        
        obs, info = self.env.reset()
        self.policy.reset()
        
        for step in range(max_steps):
            # 获取随机动作
            action = self.policy.get_action(obs)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 记录数据
            step_data = {
                'observation': self.collect_observation(obs),
                'action': action,
                'reward': float(reward),
                'terminated': terminated,
                'truncated': truncated,
            }
            traj.append(step_data)
            
            obs = next_obs
            
            # 可选：显示环境
            # self.env.render()
            
            if terminated or truncated:
                break
        
        return traj, info
    
    def save_episode(self, traj: list, episode_id: int):
        """保存单个 episode 到 HDF5"""
        filepath = self.dataset_dir / f"demo_{episode_id:04d}.h5"
        
        with h5py.File(filepath, 'w') as f:
            # 保存元数据
            f.attrs['env_name'] = self.env_name
            f.attrs['policy'] = 'random'
            f.attrs['num_steps'] = len(traj)
            f.attrs['timestamp'] = datetime.now().isoformat()
            
            # 创建数据集
            num_steps = len(traj)
            
            # 图像数据
            for key in ['image_head_rgb', 'image_left_wrist_rgb', 'image_right_wrist_rgb']:
                if key in traj[0]['observation']:
                    img_shape = traj[0]['observation'][key].shape
                    imgs = np.zeros((num_steps,) + img_shape, dtype=np.uint8)
                    for i, step in enumerate(traj):
                        imgs[i] = step['observation'].get(key, np.zeros(img_shape, dtype=np.uint8))
                    f.create_dataset(key, data=imgs, compression='gzip')
            
            # 状态数据
            for key in ['qpos', 'left_arm_joint_position', 'right_arm_joint_position',
                       'left_arm_gripper_position', 'right_arm_gripper_position',
                       'left_arm_ee_pose', 'right_arm_ee_pose']:
                if key in traj[0]['observation']:
                    dim = traj[0]['observation'][key].shape[0]
                    data = np.zeros((num_steps, dim), dtype=np.float32)
                    for i, step in enumerate(traj):
                        data[i] = step['observation'].get(key, np.zeros(dim, dtype=np.float32))
                    f.create_dataset(f'obs_{key}', data=data, compression='gzip')
            
            # 动作数据
            action_dim = traj[0]['action'].shape[0]
            actions = np.zeros((num_steps, action_dim), dtype=np.float32)
            for i, step in enumerate(traj):
                actions[i] = step['action']
            f.create_dataset('actions', data=actions, compression='gzip')
            
            # 奖励和终止标志
            rewards = np.array([step['reward'] for step in traj], dtype=np.float32)
            terminated = np.array([step['terminated'] for step in traj], dtype=bool)
            truncated = np.array([step['truncated'] for step in traj], dtype=bool)
            
            f.create_dataset('rewards', data=rewards)
            f.create_dataset('terminated', data=terminated)
            f.create_dataset('truncated', data=truncated)
        
        logger.info(f"Saved episode {episode_id} to {filepath}")
    
    def collect_demos(self, num_demos: int = 100, max_steps: int = 200):
        """收集多个演示"""
        logger.info(f"Starting to collect {num_demos} demos with random policy")
        logger.info(f"Environment: {self.env_name}")
        logger.info(f"Save directory: {self.dataset_dir}")
        
        meta_info = []
        
        while self.num_collected < num_demos:
            self.num_tries += 1
            
            # 收集 episode
            traj, info = self.collect_episode(max_steps=max_steps)
            
            # 保存数据
            self.save_episode(traj, self.num_collected)
            self.num_collected += 1
            
            # 记录元数据
            meta = {
                'episode_id': self.num_collected,
                'num_steps': len(traj),
                'num_tries': self.num_tries,
                'success_rate': f"{self.num_collected / self.num_tries * 100:.1f}%",
                'timestamp': datetime.now().isoformat()
            }
            meta_info.append(meta)
            
            # 保存元数据
            meta_file = self.dataset_dir / "meta_info.json"
            with open(meta_file, 'w') as f:
                json.dump(meta_info, f, indent=2)
            
            logger.info(f"Collected {self.num_collected}/{num_demos} demos "
                       f"(tries: {self.num_tries}, success: {meta['success_rate']})")
        
        logger.info(f"Done! Collected {num_demos} demos in {self.num_tries} tries")
    
    def close(self):
        """关闭环境"""
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Collect demos with random policy in simulation")
    parser.add_argument("--env-name", type=str, default="R1ProBlocksStackEasy-v0",
                       help="Gym environment name")
    parser.add_argument("--num-demos", type=int, default=50,
                       help="Number of demonstrations to collect")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum steps per episode")
    parser.add_argument("--dataset-dir", type=str, default="datasets",
                       help="Directory to save datasets")
    
    args = parser.parse_args()
    
    collector = DemoCollector(args.env_name, args.dataset_dir)
    
    try:
        collector.collect_demos(args.num_demos, args.max_steps)
    finally:
        collector.close()


if __name__ == "__main__":
    main()
