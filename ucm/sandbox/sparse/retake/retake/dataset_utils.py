import os
import os.path as osp
import io
import re
import json
import math
import base64
from PIL import Image
import pandas as pd
from tqdm import tqdm
from typing import Optional, List

try:
    os.environ['OPENAI_BASE_URL'] = None
    os.environ['OPENAI_API_KEY'] = None
    import openai
except:
    print("Warning! openai not installed for MLVU evalutation")
import numpy as np


class BaseDataset:
    def __init__(self, 
                 anno_file: str,
                 processor_kwargs: str
                 ) -> None:
        self.processor_kwargs = processor_kwargs
        # Load annotations
        with open(anno_file, 'r') as F:
            self.annos = json.load(F)
        # Preprocess meta
        for anno in self.annos:
            # NOTE: Pyarrow caching in LLaMA-Factory will raise error
            # for some complicate json data. So dump to jsons.
            if type(anno['meta']) == str:
                anno['meta'] = json.loads(anno['meta'])

    @staticmethod
    def _get_video_sample_extracted_frames(frame_files: List[str], **kwargs) -> int:
        video_fps = kwargs.get("video_fps")
        video_maxlen = kwargs.get("video_maxlen")
        extraction_fps = kwargs.get("video_frame_extraction_fps")
        total_frames = len(frame_files)
        sample_frames = float(total_frames / extraction_fps) * video_fps
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        sample_frames = math.floor(sample_frames)
        sampling_fps = sample_frames / (total_frames / extraction_fps)
        return int(sample_frames / 2) * 2, sampling_fps

    @staticmethod
    def _preprocess_image(image, **kwargs):
        r"""
        Pre-processes a single image.
        """
        image_resolution: int = kwargs.get("image_resolution")
        if max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def __len__(self):
        return len(self.annos)

    def get_video_message(self, video_root: str):
        frames = []
        frame_files = [
            os.path.join(video_root, file) for file in list(sorted(os.listdir(video_root)))
        ]
        total_frames = len(frame_files)
        sample_frames, sampling_fps = self._get_video_sample_extracted_frames(frame_files, **self.processor_kwargs)
        sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
        for frame_idx, frame_file in enumerate(frame_files):
            if frame_idx in sample_indices:
                # NOTE: Load and resize on the fly can creatly RAM cost of dataloader
                image = Image.open(frame_file)
                resized_image = self._preprocess_image(image, **self.processor_kwargs)
                frames.append(resized_image)

        return frames, sampling_fps

    def __getitem__(self, idx):
        anno = self.annos[idx]

        question = anno["messages"][0]["content"].replace('<video>', '')
        frames, sampling_fps = self.get_video_message(anno["videos"][0])
        meta = anno["meta"]
        meta['answer'] = anno["messages"][1]["content"]

        messages = dict(
            question=question,
            video=frames,
            sampling_fps=sampling_fps
        )

        return (idx, messages, meta)


def evaluate_mlvu_generation(anno_id, gt_answer, pred_answer, meta, enable_gpt_eval):
    """
    Evaluates question and answer pairs using GPT-4
    Returns a score for correctness.
    # Cost:
    # Before
    # 3,994.95 / ￥4,514.22

    # After
    # 4,043.27 / ￥4,514.22
    """
    if enable_gpt_eval and meta['question_type'] in ['Video Summary', 'Sub-Scene Captioning']:
        question = meta['question'].replace('<video>', '')
        pred_answer = meta['original_answer']
        client = openai.OpenAI()
        for _ in range(3):
            try:
                if meta['question_type'] == 'Video Summary':
                    response = client.chat.completions.create(
                        temperature=0,
                        model="gpt-4-turbo",
                        messages = [
                            {
                                "role": "system",
                                "content": 
                                """
                                    ##TASK DESCRIPTION: 
                                    You are required to evaluate the performance of the respondent in the video summarization task based on the standard answer and the respondent's answer. You should provide two scores. The first is the COMPLETENESS score, which should range from 1 to 5. The second is the RELIABILITY score, which should also range from 1 to 5. Below are the criteria for each scoring category:
                                    ##COMPLETENESS Scoring Criteria:
                                    The completeness score focuses on whether the summary covers all key points and main information from the video. 
                                    Score 1: The summary hardly covers any of the main content or key points of the video.
                                    Score 2: The summary covers some of the main content and key points but misses many.
                                    Score 3: The summary covers most of the main content and key points.
                                    Score 4: The summary is very comprehensive, covering most to nearly all of the main content and key points.
                                    Score 5: The summary completely covers all the main content and key points of the video.
                                    ##RELIABILITY Scoring Criteria:
                                    The reliability score evaluates the correctness and clarity of the video summary. It checks for factual errors, misleading statements, and contradictions with the video content. If the respondent's answer includes details that are not present in the standard answer, as long as these details do not conflict with the correct answer and are reasonable, points should not be deducted.
                                    Score 1: Contains multiple factual errors and contradictions; presentation is confusing.
                                    Score 2: Includes several errors and some contradictions; needs clearer presentation.
                                    Score 3: Generally accurate with minor errors; minimal contradictions; reasonably clear presentation.
                                    Score 4: Very accurate with negligible inaccuracies; no contradictions; clear and fluent presentation.
                                    Score 5: Completely accurate with no errors or contradictions; presentation is clear and easy to understand.
                                    ----
                                    ##INSTRUCTION:
                                    1. Evaluate COMPLETENESS: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence. 
                                    2. Evaluate RELIABILITY: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence. 
                                    3. Output Scores in JSON Format: Present the scores in JSON format as follows:
                                    {'score_completeness': score_comp, 'score_reliability': score_reli, 'total_score': score_comp + score_reli}
                                """
                            },
                            {
                                "role": "user",
                                "content": f"""
                                    Please score the respondent's answer according to the steps in the Instructions. You must end with a JSON dict to store the scores.
                                    Standard Answer: {gt_answer}
                                    Respondent's Answer: {pred_answer}
                                """
                            }
                        ]
                    )
                else: # "Sub-Scene Captioning"
                    scoring_points = meta['scoring_points']

                    response = client.chat.completions.create(
                        temperature=0,
                        model="gpt-4-turbo",
                        messages = [
                            {
                                "role": "system",
                                "content": 
                                """
                                    ##TASK DESCRIPTION: 
                                    You are required to evaluate a respondent's answer based on a provided question, some scoring points, and the respondent's answer. You should provide two scores. The first is the accuracy score, which should range from 1 to 5. The second is the relevance score, which should also range from 1 to 5. Below are the criteria for each scoring category.
                                    ##ACCURACY Scoring Criteria: 
                                    Evaluate the respondent's answer against specific scoring points as follows:
                                    Score 1: The response completely misses the scoring point.
                                    Score 3: The response mentions content related to the scoring point but is not entirely correct.
                                    Score 5: The response accurately addresses the scoring point.
                                    Calculate the average score across all scoring points to determine the final accuracy score.
                                    ##RELEVANCE Scoring Criteria:
                                    Assess how the respondent's answer relates to the original question:
                                    Score 1: The response is completely off-topic from the question.
                                    Score 2: The response is partially related to the question but contains a significant amount of irrelevant content.
                                    Score 3: The response primarily addresses the question, but the respondent seems uncertain about their own answer.
                                    Score 4: The response mostly addresses the question and the respondent appears confident in their answer.
                                    Score 5: The response is fully focused on addressing the question with no irrelevant content and demonstrates complete certainty.
                                    ----
                                    ##INSTRUCTION:
                                    1. Evaluate Accuracy: First, assess and score each scoring point based on the respondent's answer. Calculate the average of these scores to establish the final accuracy score. Provide a detailed rationale before assigning your score.
                                    2. Evaluate RELEVANCE: Assess the relevance of the respondent’s answer to the question. Note that when evaluating relevance, the correctness of the answer is not considered; focus solely on how relevant the answer is to the question. Provide a comprehensive rationale before assigning your score.
                                    3. Output Scores in JSON Format: Present the scores in JSON format as follows:
                                    {'score_accuracy': score_acc, 'score_relevance': score_rele, 'total_score': score_acc + score_rele}
                                """
                            },
                            {
                                "role": "user",
                                "content": f"""
                                    Please score the respondent's answer according to the steps in the Instructions. You must end with a JSON dict to store the scores.
                                    Question: {question}
                                    Scoring Points: {scoring_points}
                                    Respondent's Answer: {pred_answer}
                                """
                            }
                        ]
                    )
                gpt_message = response.choices[0].message.content
                # Use regex to extract the JSON part of the string
                json_match = re.search(r'```json\n(.*?)\n```', gpt_message, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    # Parse the JSON string into a Python dictionary
                    json_dict = json.loads(json_str)
                    score = json_dict['total_score']
                    break
                else:
                    print(f"No JSON found in the string. {anno_id}")
                    score = 0
            except Exception as e:
                score = 0
                print(f"Error processing anno_id '{anno_id}': {e}")
    else:
        if gt_answer.lower() == pred_answer.lower():
            score = 1
        else:
            score = 0
        gpt_message = ""

    return score, gpt_message


def eval_videomme_results(anno_id2result, anno_id2meta, **kwargs):
    # Load and merge all result files
    anno_id_list, subfield_list, domain_list, duration_list = [], [], [], []
    gt_answer_list, pred_answer_list, correct_list = [], [], []
    for anno_id in anno_id2result:
        pred_answer = anno_id2result[anno_id]
        meta = anno_id2meta[anno_id]
        gt_answer = meta['answer']

        anno_id_list.append(anno_id)
        subfield_list.append(meta['task_type'])
        domain_list.append(meta['domain'])
        duration_list.append(meta['duration'])
        gt_answer_list.append(gt_answer)
        pred_answer_list.append(pred_answer)
        if gt_answer.lower() == pred_answer.lower():
            correct_list.append(1)
        else:
            correct_list.append(0)

    infer_result_df = pd.DataFrame({
        'anno_id': anno_id_list,
        'subfield': subfield_list,
        'domain': domain_list,
        'duration': duration_list,
        'gt_answer': gt_answer_list,
        'pred_answer': pred_answer_list,
        'correct': correct_list
    })

    # Evaluation
    # Calculate accuracy per subfield
    subfield_accuracy = infer_result_df.groupby('subfield')['correct'].mean()

    # Calculate accuracy per duration
    duration_accuracy = infer_result_df.groupby('duration')['correct'].mean()

    duration_subfield_accuracy = infer_result_df.groupby(['duration', 'subfield'])['correct'].mean()
    final_df = duration_subfield_accuracy.unstack()

    # Overall agregated in duration
    final_df.loc[len(final_df)] = subfield_accuracy
    final_df.index.values[-1] = 'overall'

    # Overall agregated in subfield
    duration_accuracy.loc[3] = duration_accuracy.mean() # NOTE: This is correct because they have the same number of samples
    duration_accuracy.index.values[-1] = 'overall'
    final_df.insert(0, 'overall', duration_accuracy)

    # Reindex the DataFrame
    new_order = ['short', 'medium', 'long', 'overall']
    eval_result_df = final_df.reindex(new_order)
    eval_result_df *= 100 # to percent
    print(eval_result_df.head())

    return eval_result_df, infer_result_df


def eval_mlvu_results(anno_id2result, anno_id2meta, enable_gpt_eval=False):
    # Load and merge all result files
    anno_id_list, question_type_list = [], []
    gt_answer_list, pred_answer_list, correct_list, gpt_message_list = [], [], [], []
    for anno_id in tqdm(anno_id2result, total=len(anno_id2result)):
        meta = anno_id2meta[anno_id]
        pred_answer = anno_id2result[anno_id]
        gt_answer = meta['answer']

        anno_id_list.append(anno_id)
        question_type_list.append(meta['question_type'])
        gt_answer_list.append(gt_answer)
        pred_answer_list.append(pred_answer)
        correct, gpt_message = evaluate_mlvu_generation(anno_id, gt_answer, pred_answer, meta, enable_gpt_eval)
        correct_list.append(correct)
        gpt_message_list.append(gpt_message)
    infer_result_df = pd.DataFrame({
        'anno_id': anno_id_list,
        'question_type': question_type_list,
        'gt_answer': gt_answer_list,
        'pred_answer': pred_answer_list,
        'correct': correct_list,
        'gpt_message': gpt_message_list,
    })

    # Judge the MLVU split
    question_types = set(infer_result_df['question_type'].tolist())
    if len(question_types) == 9:
        split = 'dev'
    elif len(question_types) == 11:
        split = 'test'
    else:
        raise NotImplementedError

    # Evaluation

    # Calculate accuracy for each 'question_type'
    accuracy_by_question_type = infer_result_df.groupby('question_type')['correct'].mean() * 100
    accuracy_by_question_type = accuracy_by_question_type.reset_index()
    accuracy_by_question_type.rename(columns={'correct': 'Accuracy'}, inplace=True)

    # Calculate AVG
    if split == 'dev':
        # TR	AR	NQA	ER	PQA	    AO	AC              SSC VS
        mc_types = ['Topic Reasoning', 'Anomaly Recognition', 'Needle QA', 
                    'Ego Reasoning', 'Plot QA', 'Action Order', 'Action Count']
    else:
        # TR	AR	NQA	ER	PQA	SQA	AO	AC	TQA	M-AVG	SSC	VS	G-Avg
        # SportsQA
        raise NotImplementedError
    g_types = ['Video Summary', 'Sub-Scene Captioning']

    # Calculate accuracy among all samples
    mc_rows = infer_result_df['question_type'].isin(mc_types)
    mc_accuracy_sample = infer_result_df[mc_rows]['correct'].mean() * 100
    g_rows = infer_result_df['question_type'].isin(g_types)
    g_accuracy_sample = infer_result_df[g_rows]['correct'].mean()

    # Calculate accuracy among all types
    mc_rows = accuracy_by_question_type['question_type'].isin(mc_types)
    mc_accuracy = accuracy_by_question_type[mc_rows]['Accuracy'].mean()
    
    g_rows = accuracy_by_question_type['question_type'].isin(g_types)
    accuracy_by_question_type.loc[g_rows, 'Accuracy'] = accuracy_by_question_type.loc[g_rows, 'Accuracy'] / 100
    g_accuracy = accuracy_by_question_type[g_rows]['Accuracy'].mean()

    # Add the overall accuracy to the DataFrame
    overall_df = pd.DataFrame({
        'question_type': ['M-AVG', 'M-AVG-sample', 'G-AVG', 'G-AVG-sample'], 
        'Accuracy': [mc_accuracy, mc_accuracy_sample, g_accuracy, g_accuracy_sample]
    })

    # Combine the results
    eval_result_df = pd.concat([accuracy_by_question_type, overall_df], ignore_index=True)
    eval_result_df = eval_result_df.set_index('question_type').transpose()

    if split == 'dev':
        new_order = ['Topic Reasoning', 'Anomaly Recognition', 
                     'Needle QA', 'Ego Reasoning', 'Plot QA',
                     'Action Order', 'Action Count', 'M-AVG', 'M-AVG-sample',
                     'Video Summary', 'Sub-Scene Captioning', 'G-AVG', 'G-AVG-sample']
    else:
        # TR	AR	NQA	ER	PQA	SQA	AO	AC	TQA	M-AVG	SSC	VS	G-Avg
        # SportsQA
        raise NotImplementedError
    eval_result_df = eval_result_df[new_order]

    print(eval_result_df.head())

    return eval_result_df, infer_result_df


def eval_lvbench_results(anno_id2result, anno_id2meta, **kwargs):
    type2correct_list = {}
    anno_id_list = []
    question_type_list = []
    gt_answer_list = []
    pred_answer_list = []
    infer_result_correct_list = []
    for anno_id in anno_id2result:
        pred_answer = anno_id2result[anno_id]
        meta = anno_id2meta[anno_id]
        gt_answer = meta['answer']
        if gt_answer.lower() == pred_answer.lower():
            correct = 1
        else:
            correct = 0

        anno_id_list.append(anno_id)
        question_type_list.append(json.dumps(meta['question_type']))
        gt_answer_list.append(gt_answer)
        pred_answer_list.append(pred_answer)
        infer_result_correct_list.append(correct)

        for question_type in meta['question_type'] + ['overall']:
            correct_list = type2correct_list.get(question_type, [])
            correct_list.append(correct)
            type2correct_list[question_type] = correct_list

    infer_result_df = pd.DataFrame({
        'anno_id': anno_id_list,
        'question_type_list': question_type_list,
        'gt_answer': gt_answer_list,
        'pred_answer': pred_answer_list,
        'correct': infer_result_correct_list
    })

    for qtype, correct_list in type2correct_list.items():
        type2correct_list[qtype] = [sum(correct_list) / len(correct_list)]
    # type2correct_list['overall'] = sum([v[0] for v in type2correct_list.values()]) / len(type2correct_list)

    eval_result_df = pd.DataFrame(type2correct_list)

    # Reindex the DataFrame
    new_order = ['entity recognition', 'event understanding', 'key information retrieval', 'temporal grounding', 'reasoning', 'summarization', 'overall']
    eval_result_df = eval_result_df[new_order]
    eval_result_df *= 100 # to percent
    print(eval_result_df.head())

    return eval_result_df, infer_result_df


def eval_longvideobench_results(anno_id2result, anno_id2meta, **kwargs):
    # Load and merge all result files
    anno_id_list, question_category_list, duration_group_list = [], [], []
    gt_answer_list, pred_answer_list, correct_list = [], [], []
    for anno_id in anno_id2result:
        pred_answer = anno_id2result[anno_id]
        meta = anno_id2meta[anno_id]
        gt_answer = meta['answer']

        anno_id_list.append(anno_id)
        question_category_list.append(meta['question_category'])
        duration_group_list.append(meta['duration_group'])
        gt_answer_list.append(gt_answer)
        pred_answer_list.append(pred_answer)
        if gt_answer.lower() == pred_answer.lower():
            correct_list.append(1)
        else:
            correct_list.append(0)

    infer_result_df = pd.DataFrame({
        'anno_id': anno_id_list,
        'question_category': question_category_list,
        'duration_group': duration_group_list,
        'gt_answer': gt_answer_list,
        'pred_answer': pred_answer_list,
        'correct': correct_list
    })

    # Evaluation
    # Calculate accuracy per question_category
    question_category_accuracy = infer_result_df.groupby('question_category')['correct'].mean()

    # Calculate accuracy per duration_group
    duration_group_accuracy = infer_result_df.groupby('duration_group')['correct'].mean()

    duration_group_question_category_accuracy = infer_result_df.groupby(['duration_group', 'question_category'])['correct'].mean()
    final_df = duration_group_question_category_accuracy.unstack()

    # Overall agregated in duration_group
    final_df.loc[len(final_df)] = question_category_accuracy
    final_df.index.values[-1] = 'overall'

    # Overall agregated in question_category
    duration_group_accuracy.loc[3] = infer_result_df['correct'].mean() #question_category_accuracy.mean()
    duration_group_accuracy.index.values[-1] = 'overall'
    final_df.insert(0, 'overall', duration_group_accuracy)

    # Reindex the DataFrame
    new_order = ['8s-15s', '15s-60s', '180s-600s', '900s-3600s', 'overall']
    eval_result_df = final_df.reindex(new_order)
    eval_result_df *= 100 # to percent
    print(eval_result_df.head())

    return eval_result_df, infer_result_df


def get_dataset(dataset_name, anno_file, processor_kwargs):
    if dataset_name.lower() in ['videomme', 'mlvu', 'lvbench', 'longvideobench']:
        return BaseDataset(anno_file, processor_kwargs)
    else:
        print("Error! Dataset not implemented!", dataset_name)
        raise NotImplementedError


def get_eval_methods(dataset_name):
    if dataset_name.lower() == 'videomme':
        return eval_videomme_results
    elif dataset_name.lower() == 'mlvu':
        return eval_mlvu_results
    elif dataset_name.lower() == 'lvbench':
        return eval_lvbench_results
    elif dataset_name.lower() == 'longvideobench':
        return eval_longvideobench_results
    else:
        print("Error! Evaluation method not implemented!", dataset_name)
        raise NotImplementedError