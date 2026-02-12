#!/usr/bin/env python
# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate fine-tuned VLA policies on Franka-Factory-MCXCardBlockInsert task.

Supports:
- Pi-Zero (LeRobot): tshiamor/pizero-mcx-card
- GR00T N1.5 (LeRobot): tshiamor/groot-n15-mcx-card
- GR00T N1.6 (Isaac-GR00T): /home/tshiamo/groot_data/finetune_output_n16
- OpenVLA: tshiamor/openvla-mcx-card

Usage:
    # Pi-Zero
    ./isaaclab.sh -p scripts/eval/eval_vla_policy.py --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
        --policy pizero --model tshiamor/pizero-mcx-card --enable_cameras

    # GR00T N1.5
    ./isaaclab.sh -p scripts/eval/eval_vla_policy.py --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
        --policy groot --model tshiamor/groot-n15-mcx-card --enable_cameras

    # GR00T N1.6 (fine-tuned)
    ./isaaclab.sh -p scripts/eval/eval_vla_policy.py --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
        --policy groot_n16 --model /home/tshiamo/groot_data/finetune_output_n16 --enable_cameras

    # OpenVLA
    ./isaaclab.sh -p scripts/eval/eval_vla_policy.py --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
        --policy openvla --model tshiamor/openvla-mcx-card --enable_cameras
"""

import argparse
from isaaclab.app import AppLauncher

# CLI arguments
parser = argparse.ArgumentParser(description="Evaluate VLA policies on MCX Card task")
parser.add_argument("--task", type=str, required=True, help="Task name")
parser.add_argument("--policy", type=str, required=True, choices=["pizero", "groot", "groot_n16", "openvla"])
parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
parser.add_argument(
    "--task_instruction",
    type=str,
    default="pick up the blue block and place it in the first card's closest slot",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after simulator launch
import gymnasium as gym
import torch
import numpy as np
from PIL import Image

import franka_factory  # noqa: F401


class VLAPolicyWrapper:
    """Wrapper for VLA policies to provide unified interface."""

    def __init__(self, policy_type: str, model_id: str, device: str = "cuda"):
        self.policy_type = policy_type
        self.model_id = model_id
        self.device = device
        self.policy = None
        self.processor = None

        self._load_policy()

    def _load_policy(self):
        """Load policy from HuggingFace."""
        print(f"Loading {self.policy_type} policy from {self.model_id}...")

        if self.policy_type == "pizero":
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            from transformers import AutoTokenizer
            self.policy = PI0Policy.from_pretrained(self.model_id)
            self.policy.to(self.device)
            self.policy.eval()
            # Load tokenizer for language
            self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        elif self.policy_type == "groot":
            from lerobot.policies.groot.modeling_groot import GrootPolicy
            from lerobot.processor.pipeline import DataProcessorPipeline
            self.policy = GrootPolicy.from_pretrained(
                pretrained_name_or_path=self.model_id,
                strict=False,
            )
            self.policy.to(self.device)
            self.policy.config.device = self.device
            self.policy.eval()
            # Load preprocessor/postprocessor with trained dataset stats from model repo
            self.preprocessor = DataProcessorPipeline.from_pretrained(
                self.model_id, config_filename="policy_preprocessor.json"
            )
            self.postprocessor = DataProcessorPipeline.from_pretrained(
                self.model_id, config_filename="policy_postprocessor.json"
            )

        elif self.policy_type == "groot_n16":
            import sys
            import importlib
            from pathlib import Path

            # Add Isaac-GR00T to path (not pip-installed, local repo)
            sys.path.insert(0, "/home/tshiamo/Isaac-GR00T")
            from gr00t.policy.gr00t_policy import Gr00tPolicy as Gr00tN16Policy
            from gr00t.data.embodiment_tags import EmbodimentTag

            # Register our modality config so the processor knows our embodiment
            config_path = "/home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/groot_finetune/franka_mcx_config.py"
            sys.path.append(str(Path(config_path).parent))
            importlib.import_module("franka_mcx_config")

            self.policy = Gr00tN16Policy(
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                model_path=self.model_id,
                device=self.device,
                strict=False,
            )

        elif self.policy_type == "openvla":
            from transformers import AutoProcessor
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            # Load processor from base model (avoids config class mismatch issues)
            base_model = "openvla/openvla-7b"
            self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
            # Get model class directly from base model's remote code
            model_class = get_class_from_dynamic_module(
                "modeling_prismatic.OpenVLAForActionPrediction", base_model
            )
            # Load fine-tuned weights
            self.policy = model_class.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            self.policy.to(self.device)
            self.policy.eval()

            # Select unnorm_key for action denormalization
            # If only one dataset in norm_stats, use it; otherwise use bridge_orig as default
            norm_stats = self.policy.norm_stats
            if len(norm_stats) == 1:
                self.unnorm_key = next(iter(norm_stats.keys()))
            elif "bridge_orig" in norm_stats:
                self.unnorm_key = "bridge_orig"
            else:
                self.unnorm_key = next(iter(norm_stats.keys()))
            print(f"  Using unnorm_key: {self.unnorm_key}")

        print(f"Policy loaded successfully!")

    def reset(self):
        """Reset policy state (e.g., action queue for GR00T)."""
        if self.policy_type in ("groot", "groot_n16"):
            self.policy.reset()

    def get_action(self, obs: dict, task_instruction: str = None) -> torch.Tensor:
        """Get action from policy given observation.

        Args:
            obs: Dictionary with 'wrist_rgb' (H,W,C), 'state' (proprioception)
            task_instruction: Language instruction for the task

        Returns:
            action: Tensor of shape (7,) - 6D pose + 1D gripper
        """
        if self.policy_type == "pizero":
            return self._get_action_pizero(obs, task_instruction)
        elif self.policy_type == "groot":
            return self._get_action_groot(obs, task_instruction)
        elif self.policy_type == "groot_n16":
            return self._get_action_groot_n16(obs, task_instruction)
        elif self.policy_type == "openvla":
            return self._get_action_openvla(obs, task_instruction)

    def _get_action_pizero(self, obs: dict, task_instruction: str) -> torch.Tensor:
        """Get action from Pi-Zero policy."""
        # Prepare image: normalize to [-1, 1] as expected by PaliGemma
        # Image should be (B, C, H, W) with values in [-1, 1]
        wrist_rgb = obs["wrist_rgb"].astype(np.float32) / 255.0  # [0, 1]
        wrist_rgb = (wrist_rgb * 2) - 1  # [-1, 1]
        wrist_tensor = torch.from_numpy(wrist_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Prepare state
        state_tensor = torch.from_numpy(obs["state"]).unsqueeze(0).float().to(self.device)

        # Tokenize language instruction (add newline as required by PaliGemma)
        task = task_instruction if task_instruction.endswith("\n") else f"{task_instruction}\n"
        tokenized = self.tokenizer(
            task,
            return_tensors="pt",
            padding="max_length",
            max_length=self.policy.config.tokenizer_max_length,
            truncation=True,
        )

        # Prepare observation in the format expected by Pi-Zero
        lerobot_obs = {
            "observation.images.wrist_rgb": wrist_tensor,
            "observation.state": state_tensor,
            "observation.language.tokens": tokenized["input_ids"].to(self.device),
            "observation.language.attention_mask": tokenized["attention_mask"].bool().to(self.device),
        }

        with torch.no_grad():
            action = self.policy.select_action(lerobot_obs)

        return action.squeeze(0)

    def _get_action_groot(self, obs: dict, task_instruction: str) -> torch.Tensor:
        """Get action from GR00T N1.5 policy.

        Uses the LeRobot preprocessor -> select_action -> postprocessor pipeline.
        The preprocessor handles:
        1. Image conversion to uint8 numpy (B, T, V, C, H, W) for Eagle VLM
        2. State padding to max_state_dim with mask
        3. Eagle VLM encoding (tokenization + image processing)
        4. Moving tensors to device
        The postprocessor handles:
        5. Selecting last timestep from action horizon
        6. Slicing to env action dimension
        7. Inverse min-max normalization (if stats provided)
        """
        from copy import deepcopy

        # Prepare image as torch tensor (1, 3, H, W) float32 in [0, 1]
        wrist_rgb = obs["wrist_rgb"].astype(np.float32) / 255.0  # (H, W, 3) in [0, 1]
        wrist_tensor = torch.from_numpy(wrist_rgb).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)

        # Prepare state as torch tensor (1, D)
        state_tensor = torch.from_numpy(obs["state"]).unsqueeze(0).float()  # (1, 18)

        # Build batch in LeRobot format
        batch = {
            "observation.images.wrist_rgb": wrist_tensor,
            "observation.state": state_tensor,
            "task": [task_instruction],
        }

        # Run through preprocessor (handles Eagle encoding, state padding, device transfer)
        processed = self.preprocessor(deepcopy(batch))

        with torch.no_grad():
            action = self.policy.select_action(processed)

        # Run through postprocessor (unnormalization, slice to env action dim)
        # Postprocessor expects a batch dict with "action" key, returns batch dict
        action_batch = self.postprocessor({"action": action})
        action = action_batch["action"]

        return action.squeeze(0)

    def _get_action_groot_n16(self, obs: dict, task_instruction: str) -> torch.Tensor:
        """Get action from GR00T N1.6 policy.

        Uses Isaac-GR00T's native Gr00tPolicy which handles:
        1. VLA processor (Eagle VLM encoding, state normalization)
        2. Diffusion action head inference (4 denoising steps)
        3. Action decoding and unnormalization
        """
        # Build observation in Gr00tPolicy's expected nested dict format
        observation = {
            "video": {
                "wrist": obs["wrist_rgb"][np.newaxis, np.newaxis, ...],   # (1,1,H,W,3) uint8
                "table": obs["table_rgb"][np.newaxis, np.newaxis, ...],   # (1,1,H,W,3) uint8
            },
            "state": {
                "eef_pos": obs["eef_pos"][np.newaxis, np.newaxis, :],     # (1,1,3) float32
                "eef_quat": obs["eef_quat"][np.newaxis, np.newaxis, :],   # (1,1,4) float32
                "gripper": obs["gripper_pos"][np.newaxis, np.newaxis, :],  # (1,1,1) float32
            },
            "language": {
                "annotation.human.task_description": [[task_instruction]],  # list[list[str]]
            },
        }

        # Gr00tPolicy returns: {"arm": (1,T,6), "gripper": (1,T,1)} where T=action_horizon
        action_dict, _ = self.policy.get_action(observation)

        # Take first timestep action, concatenate arm + gripper → 7D
        arm = action_dict["arm"][0, 0, :]        # (6,)
        gripper = action_dict["gripper"][0, 0, :]  # (1,)
        action = np.concatenate([arm, gripper])    # (7,)

        return torch.from_numpy(action).float().to(self.device)

    def _get_action_openvla(self, obs: dict, task_instruction: str) -> torch.Tensor:
        """Get action from OpenVLA policy.

        Uses the model's predict_action() method which properly handles:
        1. Action token generation
        2. Token-to-bin decoding (vocab_size - token_id)
        3. Bin-to-continuous mapping via bin_centers
        4. Unnormalization using dataset norm_stats
        """
        # Convert to PIL Image
        image = Image.fromarray(obs["wrist_rgb"].astype(np.uint8))

        # Format prompt for OpenVLA: "In: What action should the robot take to {instruction}?\nOut:"
        prompt = f"In: What action should the robot take to {task_instruction}?\nOut:"

        # Process inputs to get input_ids
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            # Use predict_action which handles action token decoding and unnormalization
            action = self.policy.predict_action(
                input_ids=inputs["input_ids"],
                unnorm_key=self.unnorm_key,
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
            )

        return torch.from_numpy(action).float().to(self.device)


def prepare_observation(env_obs: dict, device: str = "cuda") -> dict:
    """Convert Isaac Lab observation to VLA policy format.

    Isaac Lab obs structure:
        policy.joint_pos: (N, 9) joint positions
        policy.joint_vel: (N, 9) joint velocities
        policy.eef_pos: (N, 3) end-effector position
        policy.eef_quat: (N, 4) end-effector quaternion
        policy.gripper_pos: (N, 2) gripper finger positions
        policy.wrist_cam: (N, H, W, 3) wrist camera RGB
        policy.table_cam: (N, H, W, 3) table camera RGB

    Returns dict with:
        wrist_rgb: (H, W, 3) uint8 RGB image
        table_rgb: (H, W, 3) uint8 RGB image
        state: (18,) proprioceptive state [joint_pos(9), joint_vel(9)]
        eef_pos: (3,) float32 end-effector position
        eef_quat: (4,) float32 end-effector quaternion
        gripper_pos: (1,) float32 gripper opening (first finger)
    """
    policy_obs = env_obs["policy"]

    # Get camera images (first env)
    wrist_rgb = policy_obs["wrist_cam"][0].cpu().numpy()  # (H, W, 3)
    if wrist_rgb.dtype != np.uint8:
        wrist_rgb = (wrist_rgb * 255).astype(np.uint8)

    table_rgb = policy_obs["table_cam"][0].cpu().numpy()  # (H, W, 3)
    if table_rgb.dtype != np.uint8:
        table_rgb = (table_rgb * 255).astype(np.uint8)

    # Construct proprioceptive state (for Pi-Zero, GR00T N1.5, OpenVLA)
    joint_pos = policy_obs["joint_pos"][0].cpu().numpy()
    joint_vel = policy_obs["joint_vel"][0].cpu().numpy()
    state = np.concatenate([joint_pos, joint_vel]).astype(np.float32)

    # Individual state components (for GR00T N1.6)
    eef_pos = policy_obs["eef_pos"][0].cpu().numpy().astype(np.float32)      # (3,)
    eef_quat = policy_obs["eef_quat"][0].cpu().numpy().astype(np.float32)    # (4,)
    gripper_pos = policy_obs["gripper_pos"][0, :1].cpu().numpy().astype(np.float32)  # (1,) first finger

    return {
        "wrist_rgb": wrist_rgb,
        "table_rgb": table_rgb,
        "state": state,
        "eef_pos": eef_pos,
        "eef_quat": eef_quat,
        "gripper_pos": gripper_pos,
    }


def run_evaluation():
    """Main evaluation loop."""
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    # Parse environment config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Create environment
    print(f"Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Load VLA policy
    vla_policy = VLAPolicyWrapper(
        policy_type=args_cli.policy,
        model_id=args_cli.model,
        device=args_cli.device,
    )

    # Run evaluation episodes
    successes = []
    rewards_all = []

    print(f"\nStarting evaluation: {args_cli.episodes} episodes, max {args_cli.max_steps} steps")
    print(f"Task: {args_cli.task_instruction}\n")

    for ep in range(args_cli.episodes):
        # Reset environment and policy state
        obs, info = env.reset()
        vla_policy.reset()
        episode_reward = 0.0
        success = False

        for step in range(args_cli.max_steps):
            # Prepare observation for VLA policy
            vla_obs = prepare_observation(obs, args_cli.device)

            # Get action from policy
            action = vla_policy.get_action(vla_obs, args_cli.task_instruction)

            # Ensure action has correct shape for environment
            # Environment expects: arm_action (6D pose) + gripper_action (1D binary)
            if action.dim() == 1:
                action = action.unsqueeze(0)  # Add batch dimension

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward.mean().item()

            if terminated.any() or truncated.any():
                # Check success from info if available
                if "success" in info:
                    success = info["success"].any().item()
                break

        successes.append(success)
        rewards_all.append(episode_reward)
        print(f"Episode {ep + 1}/{args_cli.episodes}: reward={episode_reward:.2f}, success={success}")

    # Print summary
    success_rate = sum(successes) / len(successes) * 100
    avg_reward = np.mean(rewards_all)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Policy: {args_cli.policy} ({args_cli.model})")
    print(f"Task: {args_cli.task}")
    print(f"Episodes: {args_cli.episodes}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"{'='*60}")

    env.close()


def main():
    """Main entry point."""
    run_evaluation()
    simulation_app.close()


if __name__ == "__main__":
    main()
