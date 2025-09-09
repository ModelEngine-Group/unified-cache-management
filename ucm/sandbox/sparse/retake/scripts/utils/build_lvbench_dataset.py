import os
import json
import math
import pandas as pd
import argparse
import glob


def main(args):
    video_root = os.path.join(args.hf_root, 'video_25fps')

    with open(os.path.join(args.hf_root, "video_info.meta.jsonl")) as F:
        dataset = [json.loads(line) for line in F.readlines()]

    question_type_all = set()

    data = []
    for video_data in dataset:
        for anno in video_data['qa']:
            question = anno['question'].replace('\n(A)', '\nOptions:\nA.')
            question = question.replace('\n(B)', '\nB.').replace('\n(C)', '\nC.').replace('\n(D)', '\nD.')
            assert '(E)' not in question
            question = f"<video>{question}.\nAnswer with the option's letter from the given choices directly."

            d = {
                "messages": [
                {
                    "content": question,
                    "role": "user"
                },
                {
                    "content": anno['answer'],
                    "role": "assistant"
                }
                ],
                "videos": [
                    os.path.join(video_root, video_data['key'])
                ],
                "meta": {
                    "video_id": video_data['key'],
                    "uid": anno['uid'],
                    "video_type": video_data['type'],
                    "question_type": anno['question_type'],
                    "time_reference": anno['time_reference'],
                }
            }
            d['meta'].update(video_data['video_info'])
            d['meta'] = json.dumps(d['meta'])
            question_type_all.union(set(anno['question_type']))
            data.append(d)

    os.makedirs(os.path.join(args.data_root, f"lvbench"), exist_ok=True)
    with open(os.path.join(args.data_root, "lvbench", f"lvbench.json"), 'w') as F:
        json.dump(data, F, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to extract frames at a specified FPS.")
    parser.add_argument('--hf_root', type=str, required=True)
    parser.add_argument('--data_root', type=str, default="./dataset")

    args = parser.parse_args()

    main(args)