"""Modality configuration for Franka MCX Card Block Insertion task.

Defines how GR00T N1.5 interprets the state, action, video, and language
modalities for fine-tuning on the MCX card block insertion dataset.

State (8D): eef_pos(3) + eef_quat(4) + gripper(1)
Action (7D): arm_delta(6, relative EEF pose) + gripper(1, absolute binary)
Videos: wrist camera + table camera (224x224 RGB)
Language: task description annotation

Usage:
    python launch_finetune.py \
        --modality-config-path /path/to/franka_mcx_config.py
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


franka_mcx_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist", "table"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["eef_pos", "eef_quat", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["arm", "gripper"],
        action_configs=[
            # arm: 6D EEF pose delta (already relative, so use ABSOLUTE to skip conversion)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper: binary (-1 close, 1 open)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(franka_mcx_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
