import os
import json
import pandas as pd


def load_jsonl(file):
    return [json.loads(line) for line in open(file, 'r').readlines()]


predict_result_dir = './results/path_to_results'
output_file = './ReTaKe_LVBench_submission.json'

LVBENCH_ANNO_FILE = "./dataset/lvbench/lvbench.json"


################ DO NOT CHANGE ################
def create_submission_file(predict_result_dir, output_file):
    results_df = pd.read_csv(os.path.join(predict_result_dir, "eval_results.csv"))

    video_id2results = {}
    res = results_df.loc[0]
    video_id2results['KIR'] = res['key information retrieval'] / 100
    video_id2results['EU'] = res['event understanding'] / 100
    video_id2results['Sum'] = res['summarization'] / 100
    video_id2results['ER'] = res['entity recognition'] / 100
    video_id2results['Rea'] = res['reasoning'] / 100
    video_id2results['TG'] = res['temporal grounding'] / 100
    video_id2results['Overall'] = res['overall'] / 100

    with open(output_file, 'w') as f:
        json.dump(video_id2results, f, indent=2)


create_submission_file(predict_result_dir, output_file)