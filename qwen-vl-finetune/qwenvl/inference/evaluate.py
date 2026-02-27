import argparse
import numpy as np
import json
from tqdm import tqdm
import os
import re
import pickle
import torch
import multiprocessing as mp
import hashlib
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import random
from math import ceil

DATASETS = {
    'charades': {
        'video_path': '/home/share/svmd5vm0/home/scut_czy1/datasets/Charades_v1_480',
        'stride': 20,
        'max_stride_factor': 0.5,
        'splits': {
            'default': {
                #'annotation_file': '/home/share/svmd5vm0/home/scut_czy1/datasets/charades_test_small.json',
                'annotation_file': '/home/share/svmd5vm0/home/scut_czy1/TimeZero/Charades/charades_annotation/val.json',
                'pad_sec': 0.0,
            },
            'small': {
                'annotation_file': 'dataset/charades-sta/charades_test_small.json',
                'pad_sec': 0.0,
            },
            'OOD-1': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 10.0,
            },
            'OOD-2': {
                'annotation_file': 'dataset/charades-sta/charades_test.json',
                'pad_sec': 15.0,
            },
            'test-ood': {
                'annotation_file': 'dataset/charades-sta/charades_test_ood.json',
                'pad_sec': 0.0,
            },
            'novel-composition': {
                'annotation_file': 'dataset/charades-sta/novel_composition.json',
                'pad_sec': 0.0,
            },
            'novel-word': {
                'annotation_file': 'dataset/charades-sta/novel_word.json',
                'pad_sec': 0.0,
            },
        }
    },
    'activitynet': {
        'video_path': '/home/svmd5vm0/whcs-share43/public/datasets/datasets--YimuWang--ActivityNet/snapshots/1a20776262cc1fe3761588843cad26b62c2b7125/all',
        'stride': 40,
        'max_stride_factor': 1,
        'splits': {
            'default': {
                'annotation_file': '/home/share/svmd5vm0/home/scut_czy1/TimeZero/ActivityNet/activitynet_annotation/val_2.json',
                'pad_sec': 0.0,
            },
            'OOD-1': {
                'annotation_file': 'dataset/activitynet/test.json',
                'pad_sec': 30.0,
            },
            'OOD-2': {
                'annotation_file': 'dataset/activitynet/test.json',
                'pad_sec': 60.0,
            },
        }
    },
    'qvhighlight': {
        #'feature_path': '/home/svmd5vm0/whcs-share43/datasets/qvhighlights/videos',
        'video_path': '/home/svmd5vm0/whcs-share43/datasets/qvhighlights/videos',
        'stride': 50,
        'max_stride_factor': 0.5,
        'splits': {
            'default': {
                'annotation_file': '/home/share/svmd5vm0/home/scut_czy1/TimeZero/qvhighlights/qvhighlights_annotation/highlight_val.json',
                'pad_sec': 0.0,
            },
        }
    },
}

# 全局缓存变量改为进程安全
PROCESS_LOCAL_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation for training-free video temporal grounding (Multi-GPU Version)'
    )
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset.')
    #parser.add_argument('--dataset', default='activitynet', type=str, help='Specify the dataset.')
    #parser.add_argument('--dataset', default='qvhighlight', type=str, help='Specify the dataset.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split.')
    parser.add_argument("--model_base", type=str, default="/home/svmd5vm0/whcs-share43/czy_output/tmswitch3/20250719_224136/checkpoint-2700")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--result_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    return parser.parse_args()

def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

def cached_process_vision_info(messages, return_video_kwargs=False):
    global PROCESS_LOCAL_CACHE
    
    video_path = None
    for msg in messages:
        for content in msg.get('content', []):
            if isinstance(content, dict) and 'video' in content:
                video_path = content['video']
                break
    
    # 添加视频内容哈希作为缓存键的一部分，提高缓存命中率
    if video_path and os.path.exists(video_path):
        with open(video_path, 'rb') as f:
            video_hash = hashlib.md5(f.read()).hexdigest()
        cache_key = f"{video_hash}_{return_video_kwargs}"
    else:
        cache_key = f"{video_path}_{return_video_kwargs}"
    
    if cache_key in PROCESS_LOCAL_CACHE:
        return PROCESS_LOCAL_CACHE[cache_key]
    
    result = process_vision_info(messages, return_video_kwargs=return_video_kwargs)
    PROCESS_LOCAL_CACHE[cache_key] = result
    
    return result

def inference(video_path, prompt, model, processor, max_new_tokens=2048, device="cuda:0"):
    messages = [
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, 
                "total_pixels": 3584 * 28 * 28, 
                "min_pixels": 16 * 28 * 28,
                },
            ]
        },
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs, video_kwargs = cached_process_vision_info(messages, return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return output_text[0]

def parse_timestamp_output(output_string):
    matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", output_string)
    if not matches:
        answer_match = re.search(r"<answer>(.*?)</answer>", output_string)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            answer_matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", answer_content)
            if answer_matches:
                last_match = answer_matches[-1]
                return float(last_match[0]), float(last_match[2])
        return None, None

    last_match = matches[-1]
    start_time_str = last_match[0]
    end_time_str = last_match[2]
    
    try:
        start_time = float(start_time_str)
        end_time = float(end_time_str)
        return start_time, end_time
    except ValueError:
        return None, None

GROUND_TEMPLATE = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""


# GROUND_TEMPLATE = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.Understand the event and the video, and follow these steps:
# In <think> tags:Semantic Deconstruction and Key Element Extraction
#     **Raw Event Quotation**:Reproduce the event description verbatim:[{EVENT}]  
#     **Core Semantic Triplet**: 
#         - **Verb**: Identify action type 
#         - **Object**: Highlight critical targets / There may be multiple
#         - **Modifier**: Decode temporal/quantitative cues (e.g., "again" = *second occurrence*).  
# Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""



# GROUND_TEMPLATE ="""To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event. Understand the event and the video, and follow these steps:

# 1. In <think> tags:Semantic Deconstruction and Key Element Extraction
#     **Raw Event Quotation**:Reproduce the event description verbatim:[{EVENT}]  
#     **Core Semantic Triplet**: 
#         - **Verb**: Identify action type 
#         - **Object**: Highlight critical targets / There may be multiple
#         - **Modifier**: Decode temporal/quantitative cues (e.g., "again" = *second occurrence*).  
#     **Consider step by step in conjunction with Semantic**::Analysis to determine the precise time period of the event

# 2. In <answer> tags:  
#    Output time range as `"{start_time} to {end_time}"` (seconds, two decimal places)

# ### Example (person picking up the phone again)
# <think>
# Raw Event: [person picking up the phone again]
# Semantic Analysis:
# - Verb: "picking up" primarily denotes the physical action of grasping and lifting an object(here, a phone).
# - Object: "people" is the agent performing the action. 
# - Object: "phone" is the target object. 
# - Modifier: "again" means locate the second occurrence in the video.
# The task is to identify the precise time period when person picking up the phone again.
# Based on semantic analysis, the phrase "picking up the phone again" fundamentally requires two conditions: a demonstrable interruption of prior interaction to justify "again," and a complete vertical displacement from surface to ear level to satisfy "picking up." The endpoint must align with motion cessation when the phone stabilizes at ear level. 
# This action likely occurs after the individual has finished eating from the bowl and is seated at the table. The sequence of actions suggests that the person might be preparing to read or engage with the book, which could involve picking up the phone again.
# </think>  

# <answer>
# 32.00 to 39.10  
# </answer>
# """





# GROUND_TEMPLATE = """Add verbs for the following event "[EVENT]" based on the video to make it a complete sentence.
# provide the complete sentence based on the video within the <answer> </answer> tags. For example <answer>close</answer>."""

# GROUND_TEMPLATE = """Add a verb for the following event "[EVENT]" based on the video to make it a complete sentence.
# Output your thought process within the <think> </think> tags including analysis.
# provide the complete sentence based on the video within the <answer> </answer> tags. For example <answer>person close the door</answer>."""

def create_work_items(data):
    work_items = []
    for vid, ann in data.items():
        for i in range(len(ann['sentences'])):
            work_items.append({
                'vid': vid,
                'ann': ann,
                'sentence_idx': i
            })
    random.shuffle(work_items)
    print(f"Total work items created: {len(work_items)}")
    return work_items

def setup_model(model_base, device):
    print(f"Setting up model on device {device}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        use_sliding_window=True,
        attn_implementation="flash_attention_2",
        device_map=device,
        use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(model_base)
    return model, processor

def get_result_path(result_dir, proc_id, model_base):
    os.makedirs(result_dir, exist_ok=True)
    model_name = os.path.basename(model_base)
    return os.path.join(result_dir, f"results_{model_name}_gpu{proc_id}.jsonl")

def append_to_jsonl(file_path, data):
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + '\n')
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

def process_work_items(work_items, video_dir_path, model_base, device, result_dir, proc_id, resume=False):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    
    # 为每个进程创建独立的日志文件
    result_path = get_result_path(result_dir, proc_id, model_base)
    processed_items = set()
    
    # 恢复处理状态
    if resume and os.path.exists(result_path):
        with open(result_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    item_id = f"{data['vid']}_{data['sentence_idx']}"
                    processed_items.add(item_id)
                    if 'iou' in data:
                        ious.append(data['iou'])
                        recall += (thresh <= data['iou'])
                except:
                    continue
    
    model, processor = setup_model(model_base, device)
    
    item_ids = [f"{item['vid']}_{item['sentence_idx']}" for item in work_items]
    remaining_items = [(i, item) for i, (item, item_id) in enumerate(zip(work_items, item_ids)) 
                      if not resume or item_id not in processed_items]
    
    if not remaining_items:
        print(f"Process {proc_id}: All items already processed")
        return ious, recall
    
    print(f"Process {proc_id}: Processing {len(remaining_items)} out of {len(work_items)} items")
    
    pbar = tqdm(remaining_items, position=proc_id, desc=f"GPU {proc_id}")
    for idx, (_, item) in enumerate(pbar):
        vid = item['vid']
        ann = item['ann']
        sentence_idx = item['sentence_idx']
        item_id = f"{vid}_{sentence_idx}"
        
        prompt = GROUND_TEMPLATE.replace('[EVENT]', ann['sentences'][sentence_idx])
        
        # 确定视频路径
        duration = ann['duration'] if 'duration' in ann else ann['video_duration']
        video_path = None
        for ext in ['mp4', 'mkv', 'webm']:
            path = os.path.join(video_dir_path, f"{vid}.{ext}")
            if os.path.isfile(path):
                video_path = path
                break
                
        # 处理视频
        if video_path:
            try:
                ans = inference(video_path, prompt, model, processor, device=device)
                sp, ep = parse_timestamp_output(ans)
                
                result_data = {
                    'vid': vid,
                    'sentence_idx': sentence_idx,
                    'prompt': prompt,
                    'answer': ans,
                    'pred_start': sp,
                    'pred_end': ep,
                    'gt_start': ann['timestamps'][sentence_idx][0],
                    'gt_end': ann['timestamps'][sentence_idx][1],
                }
                
                if (sp is not None) and (ep is not None):
                    s, e = ann['timestamps'][sentence_idx]
                    inter = min(e, ep) - max(s, sp)
                    union = max(e, ep) - min(s, sp)
                    iou_ = inter / union if union > 0 else 0
                    iou_ = max(iou_, 0)
                    
                    ious.append(iou_)
                    recall += (thresh <= iou_)
                    result_data['iou'] = iou_
                else:
                    ious.append(0)
                    result_data['iou'] = 0
                
                # 写入JSONL日志
                append_to_jsonl(result_path, result_data)
                
                # 更新进度条
                miou = sum(ious) / len(ious) if ious else 0
                recall_str = str(recall / len(ious) if ious else [0, 0, 0])
                pbar.set_postfix({"mIoU": miou, 'recall': recall_str})
                
            except Exception as e:
                print(f"Error processing {vid}_{sentence_idx}: {e}")
    
    print(f'=== GPU {proc_id} final result ===')
    print('mIoU:', sum(ious) / len(ious) if ious else 0)
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious) if ious else 0)
                
    return ious, recall

def evaluate(data, args, proc_id):
    dataset = DATASETS[args.dataset]
    video_dir_path = dataset['video_path']
    
    work_items = create_work_items(data)
    
    ious, recall = process_work_items(
        work_items, 
        video_dir_path, 
        args.model_base, 
        f"cuda:{proc_id}", 
        args.result_dir,
        proc_id,
        args.resume
    )
    
    return ious, recall

def split_data(data, num_gpus):
    """将数据分割为多个部分供不同GPU处理"""
    if isinstance(data, dict):
        data_items = list(data.items())
    else:
        data_items = data
        
    data_size = len(data_items)
    chunk_size = ceil(data_size / num_gpus)
    chunks = [dict(data_items[i*chunk_size:(i+1)*chunk_size]) for i in range(num_gpus)]
    
    return chunks

def worker(proc_id, data_chunk, args):
    """工作进程函数，绑定到指定GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(proc_id)
    print(f"Process {proc_id} using GPU {proc_id}, handling {len(data_chunk)} samples")
    evaluate(data_chunk, args, proc_id)

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')  # 👈 关键修复点[8,9](@ref)

    args = get_args()
    assert args.dataset in DATASETS
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits']
    
    print('Evaluating', args.dataset, args.split)
    
    # 加载数据
    with open(dataset['splits'][args.split]['annotation_file']) as f:
        data = json.load(f)
    
    # 分割数据给多个GPU
    data_chunks = split_data(data, args.num_gpus)
    
    processes = []
    for proc_id in range(args.num_gpus):
        p = mp.Process(target=worker, args=(proc_id, data_chunks[proc_id], args))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    print("All processes completed.")