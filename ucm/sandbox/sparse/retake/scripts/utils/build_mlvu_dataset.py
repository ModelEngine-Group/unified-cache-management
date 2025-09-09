import os
import json
import glob
import argparse


QTYPE_FORMAT_DICT = {
    # Multiple choice tasks
    'plotQA': "Plot QA",
    'findNeedle': "Needle QA",
    'ego': "Ego Reasoning",
    'count': "Action Count",
    'order': "Action Order",
    'anomaly_reco': "Anomaly Recognition",
    'topic_reasoning': "Topic Reasoning",
    # Generation tasks
    'subPlot': 'Sub-Scene Captioning',
    'summary': 'Video Summary',
}


def main(args):
    video_root = os.path.join(args.hf_root, 'MLVU/video_25fps')

    data = []
    anno_files = glob.glob(os.path.join(args.hf_root, "MLVU/json/*.json"))
    for anno_file in anno_files:
        with open(anno_file, 'r') as F:
            raw_data = json.load(F)

        if os.path.basename(anno_file) in ['8_sub_scene.json', '9_summary.json']:
            task_category = 'G'
        else:
            task_category = 'M'

        for sample in raw_data:
            question = sample['question']

            if task_category == 'M':
                if 'candidates' not in sample:
                    print("Warning, candidates not found", anno_file)
                    continue
                candidates = sample['candidates']

                question = "<video>{question}\nOptions:\nA. {o1}.\nB. {o2}.\nC. {o3}.\nD. {o4}.\nAnswer with the option's letter from the given choices directly.".format(
                    question=question, o1=candidates[0], o2=candidates[1], o3=candidates[2], o4=candidates[3]
                )

                answer = None
                for a, cand in zip(['A', 'B', 'C', 'D'], candidates):
                    if cand == sample['answer']:
                        answer = a
                        break
                if answer is None:
                    print("Warning! Answer not found for current sample, so it is deleted:", sample)
                    continue
                scoring_points = None
            else:
                question = "<video>{question}".format(question=question)
                answer = sample['answer']
                scoring_points = sample.get('scoring_points', None)

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
                    os.path.join(video_root, os.path.splitext(sample["video"])[0])
                ],
                "meta": {
                    "video": sample["video"],
                    "duration": sample["duration"],
                    "question_type": question_type,
                }
            }
            if scoring_points is not None:
                d['meta']['scoring_points'] = scoring_points
            data.append(d)

    os.makedirs(os.path.join(args.data_root, "mlvu"), exist_ok=True)
    with open(os.path.join(args.data_root, "mlvu", "mlvu.json"), 'w') as F:
        json.dump(data, F, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to extract frames at a specified FPS.")
    parser.add_argument('--hf_root', type=str, required=True)
    parser.add_argument('--data_root', type=str, default="./dataset")

    args = parser.parse_args()

    main(args)