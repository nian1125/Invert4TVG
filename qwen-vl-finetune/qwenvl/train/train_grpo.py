import os
import torch
import transformers
import sys
import json
import random
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, DatasetDict
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    Qwen2_5_VLForConditionalGeneration
)
from qwenvl.data.data_qwen import make_grpo_data_module

from qwenvl.train.argument import GRPOScriptArguments
from qwenvl.train.reward import REWARD_FUNCS
from qwenvl.train.qwenvl_grpo_trainer import QwenVLGRPOTrainer
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Trainer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, ModelConfig, TrlParser, get_peft_config
import datetime


def load_json_dataset(train_data_path, eval_data_path, video_folder):#, preprocessed_data_path=None): # Modified to accept preprocessed_data_path
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        examples = []
        for video_id, video_data in tqdm(data.items()):
            for sentence_id, (timestamps, sentence,miss,verb) in enumerate(zip(video_data['timestamps'], video_data['sentences'],video_data['Missing'],video_data['verb'])):
                sentence = sentence.strip().lower()
                if sentence.endswith("."):
                    sentence = sentence[:-1]
                    
                if miss.endswith("."):
                    miss = miss[:-1]
                video_filename_base = video_id
                video_path = None
                video_path = os.path.join(video_folder, f"{video_filename_base}.mp4")
                if video_path is None:
                    print(f"Warning: Video file not found for ID: {video_id}")
                    continue

                example = {
                    "problem": sentence,
                    "solution": (timestamps[0], timestamps[1]),
                    "video_path": video_path,
                    "durations": video_data['duration'],
                    "miss":miss,
                    "action":video_data['action'],
                    "verb":verb
                }
                examples.append(example)

        random.shuffle(examples)
        print(len(examples))
        print(examples[:5])
        dataset = Dataset.from_list(examples)


        def __getitem__(self, idx): # Define getitem within the scope where dataset is available
            example = dataset[idx]
            data_to_return = {k: v for k, v in example.items()} # Create a copy to avoid modifying original dataset

            try:
                messages = [[{"role": "user", "content": [{"type": "video", "video": example["video_path"][i], "total_pixels": 3584 * 28 * 28, "min_pixels": 16 * 28 * 28,},]}] for i in range(len(idx))]
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                fps_inputs = video_kwargs['fps']
                data_to_return["video_inputs"] = [[video_input] for video_input in video_inputs]
                data_to_return["video_kwargs"] = [{key: [value[i]] for key, value in video_kwargs.items()} for i in range(len(idx))]
                data_to_return["use_preprocessed"] = [True] * len(idx) # Flag to indicate preprocessed data is used
            except Exception as e:
                print(f"Warning: Error loading preprocessed data from {example['video_path'][0]}, falling back to video_path. Error: {e}")
                data_to_return["use_preprocessed"] = [False] # Fallback to video_path if loading fails
                print(idx)
                idx = idx + 1
                return self.__getitem__(idx)

            return data_to_return

        dataset.__getitem__ = __getitem__.__get__(dataset, Dataset) # Bind getitem to the dataset

        return dataset

    train_dataset = create_dataset_from_json(train_data_path, "train")
    eval_dataset = create_dataset_from_json(eval_data_path, "eval")
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})


def train():
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])  
    # 解析命令行参数
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    training_args.output_dir = os.path.join(training_args.output_dir, timestamp)
    os.makedirs(training_args.output_dir, exist_ok=True)

    # 设置奖励函数
    reward_funcs = [REWARD_FUNCS[func] for func in script_args.reward_funcs]
    


    # 读取数据集
    dataset = load_json_dataset(
        script_args.train_data_path,
        script_args.eval_data_path,
        script_args.video_folder,
        # script_args.preprocessed_data_path # Pass preprocessed_data_path
    )

    # 设置训练器
    trainer = QwenVLGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # 训练
    #trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.train()

    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()