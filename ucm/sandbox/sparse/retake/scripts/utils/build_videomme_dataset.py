import os
import json
import copy
import argparse
from tqdm import tqdm
import pysubs2
import pandas as pd
from transformers import AutoTokenizer


def get_subtitles(srt_path, question, tokenizer, max_tokens):
    if srt_path and os.path.exists(srt_path):
        subs = pysubs2.load(srt_path, encoding="utf-8")
        subtitles = []
        for sub in subs:
            sub_text = sub.text.replace("\\N", " ")
            if sub_text.strip():
                subtitles.append(sub_text)
        subtitles = "\n".join(subtitles)

        question_tokenized = tokenizer(question).input_ids
        subtitles_tokenized = tokenizer(subtitles).input_ids
        if len(question_tokenized) + len(subtitles_tokenized) > max_tokens:
            cutoff = len(question_tokenized) + len(subtitles_tokenized) - max_tokens
            subtitles_tokenized = subtitles_tokenized[:-cutoff]
            subtitles = tokenizer.decode(subtitles_tokenized, skip_special_tokens=True)
    else:
        subtitles = ""

    return subtitles


def main(args):
    video_root = os.path.join(args.hf_root, 'video_25fps')
    data_root = args.data_root
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    srt_root = os.path.join(args.hf_root, "subtitle")
    annos = pd.read_parquet(
        os.path.join(args.hf_root, 'videomme', 'test-00000-of-00001.parquet')
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_qwen2vl7b_path)


    data = []
    data_sub = []
    for idx, row in tqdm(annos.iterrows(), total=len(annos)):
        question = "<video>%s\nOptions:\n%s\nAnswer with the option's letter from the given choices directly." % (
            row["question"], '\n'.join(row["options"])
        )
        d = {
            "messages": [
            {
                "content": question,
                "role": "user"
            },
            {
                "content": row["answer"],
                "role": "assistant"
            }
            ],
            "videos": [
                os.path.join(video_root, row["videoID"])
            ],
            "meta": {
                "video_id": row["video_id"],
                "question_id": row["question_id"],
                "duration": row["duration"],
                "domain": row["domain"],
                "sub_category": row["sub_category"],
                "task_type": row["task_type"],
            }
        }
        data.append(d)

        subtitles = get_subtitles(os.path.join(srt_root, f'{row["videoID"]}.srt'), question, tokenizer, args.max_tokens)
        if subtitles != "":
            question = "<video>This video's subtitles are listed below:\n%s\n%s\nOptions:\n%s\nAnswer with the option's letter from the given choices directly." % (
                subtitles, row["question"], '\n'.join(row["options"])
            )
        d = copy.deepcopy(d)
        d['messages'][0]['content'] = question
        data_sub.append(d)


    os.makedirs(os.path.join(data_root, "video_mme"), exist_ok=True)
    with open(os.path.join(data_root, "video_mme", "video_mme.json"), 'w') as F:
        json.dump(data, F, indent=2)
    with open(os.path.join(data_root, "video_mme", "video_mme_subtitle.json"), 'w') as F:
        json.dump(data_sub, F, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to extract frames at a specified FPS.")
    parser.add_argument('--hf_qwen2vl7b_path', type=str, required=True)
    parser.add_argument('--hf_root', type=str, required=True)
    parser.add_argument('--data_root', type=str, default="./dataset")
    parser.add_argument('--max_tokens', type=int, default=10000) # cover 90% long videos

    args = parser.parse_args()

    main(args)