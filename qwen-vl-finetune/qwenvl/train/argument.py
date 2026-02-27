import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from trl import ScriptArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: float = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    reward_funcs: list[str] = field(
        #default_factory=lambda: ["iou", "format","spatiotemporal_reward"],
        default_factory=lambda: ["iou", "format"],
        #default_factory=lambda: ["iou", "format","act"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )
    eval_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_folder: str = field(
        default="/share/wy/Video/Charades/Charades_v1",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )
    preprocessed_data_path: Optional[str] = field( # Add preprocessed_data_path argument
        default="",
        metadata={"help": "Path to the preprocessed dataset directory. If provided, load preprocessed data instead of raw videos."},
    )
  