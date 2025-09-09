import os
import json
import pandas as pd


predict_result_dir = 'results/path_to_results'
output_file = './ReTaKe_videomme_submission.json'


videomme_hf_root = "/Video-MME/origin_data"
data_root = "./dataset"


################ DO NOT CHANGE ################
annos = pd.read_parquet(
    os.path.join(videomme_hf_root, 'videomme', 'test-00000-of-00001.parquet')
)
with open(os.path.join(predict_result_dir, "generated_predictions.jsonl"), 'r') as f:
    responses = [json.loads(line) for line in f.readlines()]

video_id2results = {}
for idx, row in annos.iterrows():
    video_id = row['video_id']
    if video_id in video_id2results:
        video_results = video_id2results[video_id]
    else:
        video_results = dict(
            video_id=video_id,
            duration=row['duration'],
            domain=row['domain'],
            sub_category=row['sub_category'],
        )
    questions = video_results.get('questions', [])
    questions.append(dict(
        question_id=row['question_id'],
        task_type=row['task_type'],
        question=row['question'],
        options=row['options'].tolist(),
        answer=row['answer'],
        response=responses[idx]['predict'],
    ))
    video_results['questions'] = questions
    video_id2results[video_id] = video_results

submission_results = []
for video_results in video_id2results.values():
    submission_results.append(video_results)


with open(output_file, 'w') as f:
    json.dump(submission_results, f, indent=2)