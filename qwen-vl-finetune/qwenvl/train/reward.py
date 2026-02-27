import numpy as np
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import random
import spacy
import shared_config

import re
import gc
import torch
from collections import OrderedDict
from typing import Set, Tuple, List
from difflib import SequenceMatcher

class VerbRewardSystem:
    def __init__(self, enable_spacy=True):
        self.nlp = spacy.load("/home/share/svmd5vm0/home/scut_czy1/Invert4TVG/qwen-vl-finetune/en_core_web_sm-3.8.0/en_core_web_sm/en_core_web_sm-3.8.0") if enable_spacy else None
    
    def extract_verbs_from_diff(self, complete_text: str, missing_text: str) -> Set[str]:
        """通过完整句子和缺失句子的差异提取动词"""
        if not complete_text or not missing_text:
            return self.extract_verbs(complete_text) if complete_text else set()
        
        # 使用序列匹配找到差异
        complete_words = complete_text.split()
        missing_words = missing_text.split()
        
        # 找到缺失的词汇
        diff_words = []
        matcher = SequenceMatcher(None, missing_words, complete_words)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'insert':
                diff_words.extend(complete_words[j1:j2])
        
        # 从差异词汇中提取动词
        diff_verbs = set()
        if self.nlp and diff_words:
            diff_text = ' '.join(diff_words)
            doc = self.nlp(diff_text)
            diff_verbs = {token.lemma_.lower() for token in doc 
                         if token.pos_ == "VERB" and token.dep_ not in ("aux", "auxpass")}
        
        # 结合直接提取的动词
        direct_verbs = self.extract_verbs(complete_text)
        return direct_verbs.union(diff_verbs)
    
    def extract_verbs(self, text: str) -> Set[str]:
        """提取所有动词词元（lemma），使用集合去重"""
        if not text:
            return set()
            
        if self.nlp:
            doc = self.nlp(text)
            # 获取所有动词原型（忽略助动词和分词）
            verbs = {token.lemma_.lower() for token in doc 
                     if token.pos_ == "VERB" and token.dep_ not in ("aux", "auxpass")}
            return verbs
        
        # 备用方法：简单分词提取（仅当禁用spacy时使用）
        words = text.split()
        return {word.rstrip("s").rstrip("ed").rstrip("ing") for word in words 
                if any(word.endswith(suffix) for suffix in ("s", "ed", "ing"))}
    
    def get_verb_variants(self, verb: str) -> Set[str]:
        """获取动词的变体形式"""
        if not self.nlp:
            return {verb}
        
        # 创建包含动词的简单句子来获取变体
        variants = {verb}
        test_sentences = [
            f"I {verb}",
            f"He {verb}s", 
            f"They {verb}ed",
            f"She is {verb}ing"
        ]
        
        for sentence in test_sentences:
            try:
                doc = self.nlp(sentence)
                for token in doc:
                    if token.pos_ == "VERB":
                        variants.add(token.lemma_.lower())
            except:
                continue
        
        return variants
    
    def calculate_reward_v1(self, truth: str, pred: str, missing_text: str = None) -> float:
        """
        版本1：严格匹配 - 只有预测的动词是真值动词或其变体才给分
        """
        # 提取真值动词
        if missing_text:
            truth_verbs = self.extract_verbs_from_diff(truth, missing_text)
        else:
            truth_verbs = self.extract_verbs(truth)
        
        pred_verbs = self.extract_verbs(pred)
        
        # 无动词场景处理
        if not truth_verbs:
            return 1.0 if not pred_verbs else 0.0
        
        # 获取所有真值动词的变体
        all_truth_variants = set()
        for verb in truth_verbs:
            all_truth_variants.update(self.get_verb_variants(verb))
        
        # 检查预测的动词是否都是真值动词的变体
        for pred_verb in pred_verbs:
            if pred_verb not in all_truth_variants:
                return 0.0
        
        # 检查是否包含所有真值动词
        return 1.0 if truth_verbs.issubset(pred_verbs) else 0.0
    
    def calculate_reward_v2(self, truth: str, pred: str, missing_text: str = None) -> float:
        """
        版本2：包容匹配 - 只要包含真值所有动词就给满分（你当前的逻辑）
        """
        # 提取真值动词
        if missing_text:
            truth_verbs = self.extract_verbs_from_diff(truth, missing_text)
        else:
            truth_verbs = self.extract_verbs(truth)
        
        pred_verbs = self.extract_verbs(pred)
        
        # 无动词场景处理
        if not truth_verbs:
            return 1.0 if not pred_verbs else 0.0
        
        # 核心逻辑：预测必须包含真值所有动词（可多不可少）
        return 1.0 if truth_verbs.issubset(pred_verbs) else 0.0

# class VerbRewardSystem:
#     def __init__(self):
#         # 加载SpaCy模型（用于词形还原）
#         self.nlp = spacy.load("/home/share/svmd5vm0/home/scut_czy1/en_core_web_sm-3.8.0/en_core_web_sm/en_core_web_sm-3.8.0")
#         # 自定义动词变体映射表（覆盖Charades常见动作）
#         self.verb_variant_map = {
#             "closing": "close", "closed": "close", "closes": "close",
#             "opening": "open", "opened": "open", "opens": "open",
#             "sitting": "sit", "sat": "sit", "sits": "sit",
#             "holding": "hold", "held": "hold", "holds": "hold"
#         }
    
#     def normalize_verb(self, verb: str) -> str:
#         """将任意动词形式归一化为原型"""
#         verb = verb.lower().strip()
#         # 1. 优先查自定义映射表（覆盖不规则变化）
#         if verb in self.verb_variant_map:
#             return self.verb_variant_map[verb]
        
#         # 2. 使用SpaCy词形还原（处理规则变化）
#         doc = self.nlp(verb)
#         for token in doc:
#             if token.pos_ == "VERB":
#                 return token.lemma_.lower()  # 如"closing"→"close"
        
#         # 3. 非动词或无法解析时返回原词
#         return verb

#     def calculate_reward(self, truth: str, pred: str) -> float:
#         """核心逻辑：比较归一化后的动词原型"""
#         # 空值处理
#         if not truth or not pred:
#             return 0.0
        
#         # 归一化动词
#         truth_base = self.normalize_verb(truth)
#         pred_base = self.normalize_verb(pred)
        
#         # 严格匹配原型
#         return 1.0 if truth_base == pred_base else 0.0





def parse_timestamp_output(output_string):
    """Parses timestamp output, similar to the example code."""
    # 1. Find all <answer>...</answer> blocks.
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)

    if not answer_matches:
        return None  # No <answer> tags found.

    # 2. Use the content of the *last* <answer> block.
    last_answer_content = answer_matches[-1]
    print('last_answer_content:', last_answer_content)

    matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", last_answer_content, re.IGNORECASE)
    if not matches:
        return None
    last_match = matches[-1]
    start_time = float(last_match[0])
    end_time = float(last_match[2])
    return start_time, end_time

def action_reward(completions, miss, problem, use_strict_matching=False, **kwargs):
    rewards = []
    reward_system = VerbRewardSystem(enable_spacy=True)
    
    for content, missing, gt in zip(completions, miss, problem):
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL)
        answer_string = answer_match.group(1).strip() if answer_match else ""
        
        try:
            if use_strict_matching:
                reward = reward_system.calculate_reward_v1(gt, answer_string, missing)
            else:
                reward = reward_system.calculate_reward_v2(gt, answer_string, missing)
        except Exception as e:
            print(f"奖励计算错误: {e}")
            reward = 0.0
        
        rewards.append(reward)
    
    print("verb_rewards", rewards)
    return rewards

# def action_reward(completions: List[str], miss: List[str], verb: List[str], **kwargs) -> List[float]:
#     rewards = []
#     reward_system = VerbRewardSystem()
    
#     for content, _, gt in zip(completions, miss, verb):
#         # 提取<answer>标签内的内容
#         answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL)
#         pred_verb = answer_match.group(1).strip() if answer_match else ""
        
#         # 计算奖励
#         try:
#             reward = reward_system.calculate_reward(gt, pred_verb)
#         except Exception as e:
#             print(f"奖励计算错误: {e}")
#             reward = 0.0
#         rewards.append(reward)
    
#     print("verb_rewards", rewards)
#     return rewards




def iou_timestamp_reward(completions, solution, durations, **kwargs): # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    # print(completions, solution, durations)
    # contents = [completion[0]["content"] for completion in completions]
    temp=kwargs['video_kwargs'][0]['fps']
    rewards = []
    # print(completions, solution, durations, **kwargs)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, duration in zip(completions, solution, durations): # Added video_durations
        reward = 0.0
        parsed_times = parse_timestamp_output(content)
        start_time, end_time = 0, 0
        gt_start, gt_end = sol
        # s, e = gt_start / duration, gt_end / duration
        s, e = gt_start, gt_end
        if parsed_times:
            start_time, end_time = parsed_times
            from_number = start_time
            to_number = end_time

            intersection = max(0, min(to_number, e) - max(from_number, s))
            union = max(to_number, e) - min(from_number, s)
            if union > 0:
                iou = intersection / union   # 0.1 0.3

            reward = iou

        print('gt second:', gt_start, gt_end)
        print('pred second:', start_time, end_time)
        print(f"------------- {current_time} IoU reward: {reward} -------------\n")

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"Content: {content}\n")
                f.write(f"pred second: {str(start_time)}, {str(end_time)}\n")
                f.write(f"gt second: {str(gt_start)}, {str(gt_end)}\n")
                f.write(f"------------- {current_time} IoU reward: {reward} -------------\n") # Modified log message
    
    return rewards




def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    print('matches:', matches)
    return [1.0 if match else 0.0 for match in matches]


def format_reward2(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""

    pattern = re.compile(r'<answer>.*?</answer>', re.DOTALL)


    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    print('matches:', matches)
    return [1.0 if match else 0.0 for match in matches]



REWARD_FUNCS = {
    'format': format_reward,
    'format2':format_reward2,
    'iou': iou_timestamp_reward,
    'act':action_reward
    
}

if __name__ == "__main__":
    # Example usage
    completions = [
        [{"content": "<think>\nThis is a thought process\n</think>\n<timestep>\n[0.0, 10.0]\n</timestep>\n"}],
        [{"content": "<think> This is a thought process </think> <timestep>[0.0, 10.0]</timestep>"}],
        [{"content": "This is a thought process </think> <timestep>[0.0, 10.0]</timestep>"}],
        [{"content": "<think> This is a thought process<timestep>[0.0, 10.0]</timestep>"}],
        [{"content": "<think> This is a thought process </think> [0.0, 10.0]</timestep>"}],
        [{"content": "<think> This is a thought process </think> <timestep>[0.0, 10.0]"}],
        [{"content": "<think> This is a thought process </think> <timestep>[0.0, 10.0]</timestep><think> This is a thought process </think>"}],
        [{"content": "<think> This is a thought process </think> <timestep>[0.0, 10.0]</timestep><timestep>[0.0, 10.0]</timestep>"}],
        [{"content": "<think> This is a thought process </think> <timestep>[0.0, 10.0]</timestep></timestep>"}],
        [{"content": "<think> This is a thought process </think> <timestep>[0., 10.0]</timestep>"}],
        [{"content": "<think> This is a thought process </think> <timestep>[0, 10]</timestep>"}],
    ]

    answers = ["[0.0, 8.0]"] * 12
    
    print(f"format_reward_func={format_reward(completions)}")
    print(f"iou_reward_func={iou_timestamp_reward(completions, answers)}")
    