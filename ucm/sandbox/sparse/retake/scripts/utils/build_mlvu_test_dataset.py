import os
import json
import glob
from copy import deepcopy
import argparse


QTYPE_FORMAT_DICT = {
    # Multiple choice tasks
    'plotQA': "Plot QA",
    'needleQA': "Needle QA",
    'ego': "Ego Reasoning",
    'count': "Action Count",
    'order': "Action Order",
    'anomaly_reco': "Anomaly Recognition",
    'topic_reasoning': "Topic Reasoning",
    'tutorialQA': "Tutorial QA",
    'sportsQA': "Sports QA",
    # Generation tasks
    'subsummary': 'Sub-Scene Captioning',
    'summary': 'Video Summary',
}


def main(args):
    video_root = os.path.join(args.hf_root, 'MLVU_Test/video_25fps')
    data = []
    for task_category, filename in zip(['M', 'G'], ['test_multi_choice_tasks.json', 'test_generation_tasks.json']):
        anno_file = os.path.join(args.hf_root, filename)
        with open(anno_file, 'r') as F:
            raw_data = json.load(F)

        for sample in raw_data:
            question = sample['question']

            if task_category == 'M':
                if 'candidates' not in sample:
                    print("Warning, candidates not found", anno_file)
                    continue
                candidates = sample['candidates']
                question_id = sample['question_id']
                assert len(candidates) < len('ABCDEFG')

                options = deepcopy(candidates)
                options = '\n'.join([f'{letter}. {option}.' for letter, option in zip('ABCDEFG', options)])
                question = f"<video>{question}\nOptions:\n{options}\nAnswer with the option's letter from the given choices directly."
            else:
                question = f"<video>{question}".format(question=question)
                question_id = "null"
            answer = "null"
            scoring_points = "null"

            question_type = QTYPE_FORMAT_DICT[sample['question_type']]

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
                    os.path.join(video_root, sample['video'].split('.mp4')[0])
                ],
                "meta": {
                    "video": sample["video"],
                    "duration": sample["duration"],
                    "question_type": question_type,
                    "question_id": question_id,
                    "task_category": task_category
                }
            }
            if scoring_points is not None:
                d['meta']['scoring_points'] = scoring_points
            data.append(d)


    os.makedirs(os.path.join(args.data_root, "mlvu"), exist_ok=True)
    with open(os.path.join(args.data_root, "mlvu", "mlvu_test.json"), 'w') as F:
        json.dump(data, F, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to extract frames at a specified FPS.")
    parser.add_argument('--hf_root', type=str, required=True)
    parser.add_argument('--data_root', type=str, default="./dataset")

    args = parser.parse_args()

    main(args)