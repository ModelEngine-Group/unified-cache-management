import os
import re
import json
import pandas as pd


def load_jsonl(file):
    return [json.loads(line) for line in open(file, 'r').readlines()]


def trimm_results(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEFG]", s):
        return ""

    matches = re.search(r"[ABCDEFG]", s)
    if matches is None:
        return ""
    return matches[0]


predict_result_dir = './results/path_to_results'
output_file = './ReTaKe_MLVU_test_{qtype}_submission.json'

MLVU_TEST_ANNO_FILE = "./dataset/mlvu/mlvu_test.json"


################ DO NOT CHANGE ################
annotations = json.load(open(MLVU_TEST_ANNO_FILE, 'r'))
predictions = load_jsonl(os.path.join(predict_result_dir, 'generated_predictions.jsonl'))

mc_results = []
subplot_results = []
summary_results = []
for anno, pred in zip(annotations, predictions):
    meta = anno['meta']
    if meta['task_category'] == 'M':
        mc_results.append(dict(
            question_id=meta['question_id'],
            question_type=meta['question_type'],
            option=trimm_results(pred['predict'])
        ))
    elif meta['question_type'] == 'Sub-Scene Captioning':
        subplot_results.append(dict(
            video_name=meta['video'],
            Q=anno['messages'][0]['content'],
            pred=pred['predict']
        ))
    else:
        summary_results.append(dict(
            video_name=meta['video'],
            Q=anno['messages'][0]['content'],
            pred=pred['predict']
        ))

with open(output_file.format(qtype='mc'), 'w') as F:
    json.dump(mc_results, F, indent=2)
with open(output_file.format(qtype='subplot'), 'w') as F:
    json.dump(subplot_results, F, indent=2)
with open(output_file.format(qtype='summary'), 'w') as F:
    json.dump(summary_results, F, indent=2)