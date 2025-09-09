import os
import os.path as osp
import json
import pickle
import re
import yaml
import datetime
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoProcessor

from retake.monkeypatch import patch_qwen2vl, patch_qwen2vl_config
from retake.dataset_utils import get_dataset, get_eval_methods


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def trimm_results(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEFG]", s):
        return ""

    matches = re.search(r"[ABCDEFG]", s)
    if matches is None:
        return ""
    return matches[0]


class InferClient:
    def __init__(self, model_name, hf_model_path, exp_configs, device) -> None:
        self.device = device
        self.model_name = model_name if model_name is not None else exp_configs['model_name']
        self.do_sample = exp_configs.get('do_sample', False)
        self.load_model(model_name, hf_model_path, exp_configs, device)

    def load_model(self, model_name, hf_model_path, exp_configs, device):
        model_name = model_name if model_name is not None else exp_configs['model_name']
        model_name = model_name.lower().replace('-', '').replace('_', '')
        if model_name == 'qwen2vl': # QWen2VL
            from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
            from retake.monkeypatch import patch_qwen2vl, patch_qwen2vl_config
            patch_qwen2vl(exp_configs['method']) # Replace some functions of QWen2VL with those from ReTaKe
            qwen2vl_config = Qwen2VLConfig.from_pretrained(hf_model_path)
            qwen2vl_config = patch_qwen2vl_config(qwen2vl_config, exp_configs)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                hf_model_path,
                config=qwen2vl_config,
                torch_dtype=torch.bfloat16,
                attn_implementation=exp_configs.get('attn_implementation', None),
                device_map=device # "auto"
            ).eval()
            processor = AutoProcessor.from_pretrained(hf_model_path)
        elif model_name == 'qwen25vl': # QWen2_5VL
            from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
            from retake.monkeypatch import patch_qwen2_5_vl, patch_qwen2_5_vl_config
            patch_qwen2_5_vl(exp_configs['method']) # Replace some functions of QWen2VL with those from ReTaKe
            qwen2_5_vl_config = Qwen2_5_VLConfig.from_pretrained(hf_model_path)
            qwen2_5_vl_config = patch_qwen2_5_vl_config(qwen2_5_vl_config, exp_configs)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                hf_model_path,
                config=qwen2_5_vl_config,
                torch_dtype=torch.bfloat16,
                attn_implementation=exp_configs.get('attn_implementation', None),
                device_map=device # "auto"
            ).eval()
            processor = AutoProcessor.from_pretrained(hf_model_path)
        elif model_name in ['llavaonevision', 'llavavideo']: # LLaVA-OneVision, LLaVA-Video
            from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration
            from retake.monkeypatch import patch_llava_onevision, patch_llava_onevision_config
            patch_llava_onevision(exp_configs['method']) # Replace some functions of LLaVA-Video with those from ReTaKe
            llava_onevision_config = LlavaOnevisionConfig.from_pretrained(hf_model_path)
            llava_onevision_config = patch_llava_onevision_config(llava_onevision_config, exp_configs)
            processor = AutoProcessor.from_pretrained(hf_model_path) 
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                hf_model_path, 
                config=llava_onevision_config,
                torch_dtype=torch.bfloat16,
                attn_implementation=exp_configs.get('attn_implementation', None),
                device_map=device # "auto"
            ) 
        else:
            raise NotImplementedError
        self.model = model
        self.processor = processor

    def infer(self, message):
        # Prepare inputs
        video = message['video']
        sampling_fps = message['sampling_fps']
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": message['question']},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        videos_kwargs = dict(fps=sampling_fps)
        inputs = self.processor(text=[text_prompt], videos=[video], padding=True, return_tensors="pt", **videos_kwargs)
        if self.device == 'auto':
            inputs = inputs.to('cuda')
        else:
            inputs = inputs.to(self.device)
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.bfloat16)

        # Inference
        output_ids = self.model.generate(**inputs, do_sample=self.do_sample, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text = output_text[0]

        return output_text


def parse_arguments():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="ReTaKe Evaluation")
    parser.add_argument('--model_name', 
                        type=str, 
                        help="Modelname")
    parser.add_argument('--hf_path',
                        '--hf_qwen2vl7b_path', 
                        type=str, 
                        default='Qwen/Qwen2-VL-7B-Instruct', 
                        help="Path to the huggingface model")
    parser.add_argument('--config_path', 
                        type=str, 
                        default='configs/retake_videomme.yaml', 
                        help="Path to the experimental config")
    parser.add_argument('--video_frame_extraction_fps', 
                        type=int, 
                        default=25, 
                        help="Video frame extraction FPS")
    parser.add_argument('--n_gpus', 
                        type=int, 
                        default=8, 
                        help="Number of GPUs to use")
    parser.add_argument('--auto_sharding', 
                        action='store_true', 
                        help="Number of GPUs to use")
    parser.add_argument('--enable_cache', 
                        action='store_true', 
                        help="Number of GPUs to use")
    parser.add_argument('--skip_eval', 
                        action='store_true', 
                        help="Number of GPUs to use")
    parser.add_argument('--timeout', 
                        type=int, 
                        default=30, 
                        help="Timeout for waiting each GPU finish inference (in minutes).")
    args = parser.parse_args()
    return args


def data_parallel_setup(rank, world_size, timeout):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=timeout))
    torch.cuda.set_device(f'cuda:{rank}')


def cleanup_parallel_setup():
    dist.destroy_process_group()


def find_start_sample_id(rank, cache_dir):
    filenames = os.listdir(cache_dir)
    processed_sample_ids = [-1]
    for filename in filenames:
        sample_id = re.findall(f'anno_id2result_{rank}_(\d+).pkl', filename)
        if len(sample_id):
            processed_sample_ids.append(int(sample_id[0]))
    start_sample_id = max(processed_sample_ids) + 1
    return start_sample_id


def main(rank, world_size, args):
    if args.auto_sharding:
        device = 'auto'
    else:
        device = f'cuda:{rank}'
        data_parallel_setup(rank, world_size, args.timeout)

    exp_configs = load_yaml(args.config_path)

    client = InferClient(args.model_name, args.hf_path, exp_configs, device)

    dataset = get_dataset(
        dataset_name=exp_configs['dataset_name'],
        anno_file=exp_configs['anno_file'], 
        processor_kwargs=dict(
            video_fps=exp_configs['sample_fps'],
            video_maxlen=exp_configs['max_num_frames'],
            image_resolution=exp_configs['longsize_resolution'],
            video_frame_extraction_fps=args.video_frame_extraction_fps
        )
    )

    # Inference
    anno_id2result = {}
    anno_id2meta = {}
    start_sample_id = 0
    if args.enable_cache:
        cache_dir = os.path.join(exp_configs['output_dir'], 
                                 'cache', 
                                 os.path.splitext(os.path.basename(args.config_path))[0])
        os.makedirs(cache_dir, exist_ok=True)
        start_sample_id = find_start_sample_id(rank, cache_dir)
        if start_sample_id > 0:
            prev_sample_id = start_sample_id - 1
            anno_id2result_cachefile = os.path.join(cache_dir, f'anno_id2result_{rank}_{prev_sample_id}.pkl')
            anno_id2meta_cachefile = os.path.join(cache_dir, f'anno_id2meta_{rank}_{prev_sample_id}.pkl')
            print('Loading cache file:', anno_id2result_cachefile)
            anno_id2result = pickle.load(open(anno_id2result_cachefile, 'rb'))
            anno_id2meta = pickle.load(open(anno_id2meta_cachefile, 'rb'))
            print("Starting from", start_sample_id)

    # Split dataset into shards
    # Function to create a round-robin shard for a given rank
    indices = [i for i in range(len(dataset)) if i % world_size == rank and i >= start_sample_id]
    shard_dataset = Subset(dataset, indices)

    dataloader = DataLoader(shard_dataset, batch_size=None, num_workers=exp_configs['dataloader_num_workers'])

    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'rank {rank}'): # disable=rank!=0
        i = i + start_sample_id
        idx, message, meta = sample
        pred_answer = client.infer(message)
        pred_answer = trimm_results(pred_answer)
        anno_id2result[idx] = pred_answer
        anno_id2meta[idx] = meta

        if args.enable_cache and i % 10 == 0:
            anno_id2result_cachefile = os.path.join(cache_dir, f'anno_id2result_{rank}_{i}.pkl')
            anno_id2meta_cachefile = os.path.join(cache_dir, f'anno_id2meta_{rank}_{i}.pkl')
            print('Dumping cache file:', anno_id2result_cachefile)
            pickle.dump(anno_id2result, open(anno_id2result_cachefile, 'wb'))
            pickle.dump(anno_id2meta, open(anno_id2meta_cachefile, 'wb'))

    if not args.auto_sharding: # Gather results from all processes
        all_anno_id2result = [None] * world_size
        all_anno_id2meta = [None] * world_size
        dist.barrier()
        dist.all_gather_object(all_anno_id2result, anno_id2result)
        dist.all_gather_object(all_anno_id2meta, anno_id2meta)
    else:
        all_anno_id2result = [anno_id2result]
        all_anno_id2meta = [anno_id2meta]

    if rank == 0:
        # Merge results
        merged_anno_id2result = {k: v for d in all_anno_id2result for k, v in d.items()}
        merged_anno_id2meta = {k: v for d in all_anno_id2meta for k, v in d.items()}
        merged_anno_id2result = dict(sorted(merged_anno_id2result.items()))
        merged_anno_id2meta = dict(sorted(merged_anno_id2meta.items()))
        # Dump results
        os.makedirs(exp_configs['output_dir'], exist_ok=True)
        answer_file = os.path.join(exp_configs['output_dir'], "anno_id2result.json")
        meta_file = os.path.join(exp_configs['output_dir'], "anno_id2meta.json")
        with open(answer_file, 'w') as F:
            json.dump(merged_anno_id2result, F)
        with open(meta_file, 'w') as F:
            json.dump(merged_anno_id2meta, F)

        if not args.skip_eval:
            # Evaluate
            eval_func = get_eval_methods(exp_configs['dataset_name'])
            eval_result_df, infer_result_df = eval_func(merged_anno_id2result, merged_anno_id2meta)

            # Dump inference & evaluation results
            infer_res_file = os.path.join(exp_configs['output_dir'], "infer_results.csv")
            eval_res_file = os.path.join(exp_configs['output_dir'], "eval_results.csv")

            infer_result_df.to_csv(infer_res_file, index=False)
            eval_result_df.to_csv(eval_res_file, index=True)

    if not args.auto_sharding:
        cleanup_parallel_setup()


if __name__ == "__main__":
    args = parse_arguments()
    world_size = args.n_gpus
    if args.auto_sharding: # For auto model parallel through huggingface
        main(0, 1, args)
    else: # For data parallel
        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
