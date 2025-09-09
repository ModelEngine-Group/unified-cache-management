import os
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
from PIL import Image
import torch


def get_duration_span(row):
    if row['duration_group'] == 15:
        return '8s-15s'
    elif row['duration_group'] == 60:
        return '15s-60s'
    elif row['duration_group'] == 600:
        return '180s-600s'
    elif row['duration_group'] == 3600:
        return '900s-3600s'
    else:
        raise ValueError


if __name__ == "__main__":
    video_root = os.path.join(args.hf_root, 'video_25fps')

    tokenizer = AutoTokenizer.from_pretrained(args.hf_qwen2vl7b_path)
    for split, anno_file in zip(['val', 'test'], ['validation-00000-of-00001.parquet', 'test-00000-of-00001.parquet']):
        data = []
        num_tokens = []
        anno_df = pd.read_parquet(os.path.join(args.hf_root, anno_file))
        for _, row in tqdm(anno_df.iterrows(), total=len(anno_df)):
            with open(os.path.join(args.hf_root, "subtitles", row["subtitle_path"])) as f:
                subtitles = json.load(f)
            try:
                subtitles = [subtitle['line'] for subtitle in subtitles]
                subtitles = ' '.join(subtitles)
            except:
                try:
                    subtitles = [subtitle['text'] for subtitle in subtitles]
                    subtitles = ' '.join(subtitles)
                except:
                    subtitles = ''
                    print('decoding error for ', os.path.join(args.hf_root, "subtitles", row["subtitle_path"]))
            if len(subtitles.split(' ')) > 8000:
                print("Extreme long subtitle!", len(subtitles.split(' ')))
                subtitles = ' '.join(subtitles.split(' ')[:8000])

            if row['option4'] == 'N/A':
                question = "<video>Subtitles: {subtitles}\n{question}\nOptions:\nA. {o1}.\nB. {o2}.\nC. {o3}.\nD. {o4}.\nAnswer with the option's letter from the given choices directly.".format(
                    subtitles=subtitles, question=row['question'], o1=row['option0'], o2=row['option1'], o3=row['option2'], o4=row['option3']
                )
            else:
                question = "<video>Subtitles: {subtitles}\n{question}\nOptions:\nA. {o1}.\nB. {o2}.\nC. {o3}.\nD. {o4}.\nE. {o5}.\nAnswer with the option's letter from the given choices directly.".format(
                    subtitles=subtitles, question=row['question'], o1=row['option0'], o2=row['option1'], o3=row['option2'], o4=row['option3'], o5=row['option4']
                )
            num_tokens.append(
                len(tokenizer(row['question']+row['option0']+row['option1']+row['option2']+row['option3']).input_ids)
            )
            if split == 'test':
                answer = 'None'
            else:
                answer = None
                for idx, a in enumerate(['A', 'B', 'C', 'D', 'E']):
                    if idx == row['correct_choice']:
                        answer = a
                        break
                assert answer is not None

            duration_group = get_duration_span(row)

            d = {
                "messages": [
                {
                    "content": question,
                    "role": "user"
                },
                {
                    "content": answer,
                    "role": "assistant"
                }
                ],
                "videos": [
                    os.path.join(video_root, os.path.splitext(row["video_path"])[0])
                ],
                "meta": {
                    "video_id": row["video_id"],
                    "question_id": row["id"],
                    "duration_group": duration_group,
                    "duration": row["duration"],
                    "question_category": row['question_category'],
                    "topic_category": row['topic_category'],
                    "subtitle_path": row['subtitle_path'],
                    "starting_timestamp_for_subtitles": row['starting_timestamp_for_subtitles'],
                }
            }
            data.append(d)

        # Calculate and print the numbers at the specified percentiles
        for p in [10, 20, 50, 80, 90]:
            value = np.percentile(num_tokens, p)
            print(f"The {p}th percentile is {value}")
        os.makedirs(os.path.join(args.data_root, f"longvideobench"), exist_ok=True)
        with open(os.path.join(args.data_root, "longvideobench", f"longvideobench_{split}.json"), 'w') as F:
            json.dump(data, F, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to extract frames at a specified FPS.")
    parser.add_argument('--hf_root', type=str, required=True)
    parser.add_argument('--data_root', type=str, default="./dataset")
    parser.add_argument('--hf_qwen2vl7b_path', type=str, default="QWen/Qwen2-VL-72B-Instruct")

    args = parser.parse_args()

    main(args)