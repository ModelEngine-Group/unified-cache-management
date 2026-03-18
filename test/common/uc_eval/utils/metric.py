import random
import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Union

import jieba
import numpy as np
from common.uc_eval.utils.data_class import MultiTurnDialogRecord, RequestRecord

stopwords_path = Path(__file__).parent.joinpath("stopwords.txt")
ZH_PUNCTUATION = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
STOPWORDS: List[str] = [
    line.strip() for line in stopwords_path.open("r", encoding="utf-8").readlines()
]


def normalize_text(text: str) -> str:
    # Remove punctuation (CJK full-width & ASCII)
    pattern = r"[\u3000-\u303F\uFF00-\uFFEF" + re.escape(string.punctuation) + "]"
    text = re.sub(pattern, "", text)
    # Segment with jieba (precise mode) and lowercase
    words = jieba.lcut(text)
    words = [word.strip().lower() for word in words]
    # Drop stop-words
    filtered_words = [word for word in words if word not in STOPWORDS and word != ""]
    text = " ".join(map(str, filtered_words))
    return text


def normalize_zh_text(text: str) -> str:
    """For Chinese text, lower text and remove punctuation, extra whitespace"""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        all_punctuation = set(string.punctuation + ZH_PUNCTUATION)
        return "".join(ch for ch in text if ch not in all_punctuation)

    return white_space_fix(remove_punc(text.lower()))


def normalize_en_text(text: str) -> str:
    """For English, lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


class MetricClass(ABC):
    def __init__(self, record_list: List[RequestRecord | MultiTurnDialogRecord]):
        self.record_list = record_list
        self.ACCURACY_METIRC_FUNCTION_MAP: dict[str, Callable] = {
            "accuracy": self.get_accuracy,
            "bootstrap-accuracy": self.get_bootstrap_accuracy_std,
            "f1-score": self.get_f1_score,
        }

    def calculate_metric(self, metric_names: List[str]):
        """
        Get evaluation metrics
        """
        for record in self.record_list:
            expected_output, real_output = self.get_expected_and_real_output(record)

            if self.match(expected_output, real_output):
                record.is_match = True

        metric_dict = {}
        for metric in metric_names:
            metric_function = self.ACCURACY_METIRC_FUNCTION_MAP[metric]
            metric_dict[metric] = metric_function(self.record_list) * 100

        return metric_dict

    def get_expected_and_real_output(
        self, record: Union[RequestRecord, MultiTurnDialogRecord]
    ):
        expected_output = record.expected_output
        real_output = record.output_data

        if isinstance(expected_output, tuple):
            expected_output = list(expected_output)
        elif not isinstance(expected_output, list):
            expected_output = [expected_output]

        return expected_output, real_output

    def remove_tail_punc(self, output):
        all_punctuation = set(string.punctuation + ZH_PUNCTUATION)
        if output and output[-1] in all_punctuation:
            return output[:-1]

        return output

    def get_normalize_text(self, record: Union[RequestRecord, MultiTurnDialogRecord]):
        """
        Perform standardization of output data
        """
        expected_output, real_output = self.get_expected_and_real_output(record)
        real_output = self.del_chain_of_thought(record.output_data)
        language = record.case_name.split("-")[0] if record.case_name else "None"
        if language.lower() == "en":
            normalized_expected = [
                normalize_en_text(output) for output in expected_output
            ]
            normalized_real = normalize_en_text(real_output)
            expected_output = [output.split() for output in normalized_expected]
            real_output = normalized_real.split()
        elif language.lower() == "zh":
            expected_output = [
                self.get_zh_output_tokens(output) for output in expected_output
            ]
            real_output = self.get_zh_output_tokens(real_output)
        else:
            expected_output = [
                normalize_text(output).split() for output in expected_output
            ]
            real_output = normalize_text(real_output).split()

        return expected_output, real_output

    def get_zh_output_tokens(self, output):
        spilt_tokens = list(jieba.cut(output, cut_all=False))
        normalized_tokens = [normalize_zh_text(token) for token in spilt_tokens]
        tokens = [token for token in normalized_tokens if len(token) > 0]
        return tokens

    def del_chain_of_thought(
        self,
        output_data: str,
        think_start_tokens: str = "<think>",
        think_end_tokens: str = "</think>",
    ):
        """
        Delete the chain of thought from the output data
        """
        if (
            think_start_tokens not in output_data
            and think_end_tokens not in output_data
        ):
            return output_data
        elif think_start_tokens not in output_data and think_end_tokens in output_data:
            return output_data.split(think_end_tokens)[-1].strip()

        start_escaped = re.escape(think_start_tokens)
        end_escaped = re.escape(think_end_tokens)
        reason_data = re.compile(rf"{start_escaped}(.*?){end_escaped}", re.DOTALL)
        return reason_data.sub("", output_data).strip()

    @abstractmethod
    def match(
        self,
        expected_output: Union[str, List[str], tuple[str]],
        real_output: str,
        **kwargs,
    ):
        pass

    def get_accuracy(
        self, record_list: List[RequestRecord | MultiTurnDialogRecord]
    ) -> float:
        record_total = len(record_list)
        match_num = sum(record.is_match for record in record_list)
        return match_num / record_total if record_total != 0 else float("nan")

    def get_bootstrap_accuracy_std(
        self,
        record_list: List[RequestRecord | MultiTurnDialogRecord],
        num_samples: int = 1000,
    ):
        """
        Compute standard deviation of accuracy using the Bootstrap method.
        """
        if not record_list:
            return float("nan")

        vals = [record.is_match for record in record_list]
        return np.std(
            [np.mean(random.sample(vals, len(vals) // 2)) for _ in range(num_samples)]
        ).item()

    def get_f1_score(
        self,
        record_list: List[RequestRecord | MultiTurnDialogRecord],
    ):
        f1_score = []
        for record in record_list:
            expected_output, real_output = self.get_normalize_text(record)
            f1_score.append(self._f1_score(expected_output, real_output))
        return np.mean(f1_score).item()

    def _f1_score(self, expected_output: List[str], real_output: str) -> float:
        max_f1_score = 0
        for output in expected_output:
            common = Counter(output) & Counter(real_output)
            num_same = sum(common.values())
            if num_same != 0:
                precision = 1.0 * num_same / len(output)
                recall = 1.0 * num_same / len(real_output)
                f1 = (2 * precision * recall) / (precision + recall)
                max_f1_score = max(max_f1_score, f1)
        return max_f1_score


class Match(MetricClass):
    def __init__(self, record_list: List[RequestRecord | MultiTurnDialogRecord]):
        super().__init__(record_list)

    def match(
        self,
        expected_output: List[str],
        real_output: str,
        separator: Callable[[str], bool] = None,
        options: Optional[list[str]] = None,
    ) -> bool:
        """
        Exact match: expected and picked must be identical
        :param expected_output: the answer from dataset
        :param real_output: actual output generated by model
        :param separator: separator function to prevent partial matches
        :param options: optional list of matching options; for multiple-choice questions, options must be present
        """
        if options is None:
            options = expected_output

        picked = None
        real_output = self.del_chain_of_thought(real_output)
        for option in options:
            if not real_output.startswith(option):
                continue
            if (
                separator is not None
                and len(real_output) > len(options)
                and not separator(real_output[len(option)])
            ):
                continue
            picked = option
            break

        match = picked in expected_output
        return match


class Includes(MetricClass):
    def __init__(self, record_list: List[RequestRecord | MultiTurnDialogRecord]):
        super().__init__(record_list)

    def match(
        self,
        expected_output: List[str],
        real_output: str,
    ) -> bool:
        """
        Match succeeds if any part expected_output is found in real_output
        :param expected_output: the answer from dataset
        :param real_output: actual output generated by model
        """
        real_output = self.del_chain_of_thought(real_output)
        for output in expected_output:
            format_output = self.remove_tail_punc(output)
            if real_output.rfind(format_output) != -1:
                return True
        return False


class FuzzyMatch(MetricClass):
    def __init__(self, record_list: List[RequestRecord | MultiTurnDialogRecord]):
        super().__init__(record_list)

    def match(
        self,
        expected_output: List[str],
        real_output: str,
        strategy: str = "substring",
        threshold: float = 0.8,
    ) -> bool:
        """
        Fuzzy matching
        :param expected_output: the answer from dataset
        :param real_output: actual output generated by model
        :param strategy: matching strategy, currently supports substring and jaccard
        :param threshold: similarity threshold for jaccard strategy
        """
        real_output = self.del_chain_of_thought(real_output)
        return any(
            self._single_match(
                self.remove_tail_punc(expected),
                self.remove_tail_punc(real_output),
                strategy,
                threshold,
            )
            for expected in expected_output
        )

    def _single_match(
        self,
        expected: str,
        real: str,
        strategy: str = "substring",
        threshold: float = 0.8,
    ) -> bool:
        if strategy == "substring":
            return expected in real or real in expected
        else:
            set_exp, set_real = set(expected.split()), set(real.split())
            if not set_exp and not set_real:
                return True
            if not set_exp or not set_real:
                return False
            inter = len(set_exp & set_real)
            union = len(set_exp | set_real)
            return (inter / union) >= threshold


class MatchPatterns(MetricClass):
    def __init__(self, record_list: List[RequestRecord | MultiTurnDialogRecord]):
        super().__init__(record_list)

    def match(
        self,
        expected_output: List[str],
        real_output: str,
    ) -> bool:
        """
        Use the provided regular expression list to extract output from output, and then judge whether it matches
        :param expected_output: the answer from dataset
        :param real_output: actual output generated by model
        """
        pred = self.get_answer_from_match_patterns(real_output)
        if pred and pred in expected_output:
            return True

        return False

    def get_answer_from_match_patterns(self, real_output: str):
        """
        Get the answer through comparing match_patterns and output
        """
        from common.uc_eval.utils.prompt_config import match_patterns_longbench_v2

        output = real_output.replace("*", "")
        for pattern in match_patterns_longbench_v2:
            match = re.search(pattern, output)
            if match:
                return match.group(1)

        return None


class GSM8KMatchPatterns(MetricClass):
    def __init__(self, record_list: List[RequestRecord | MultiTurnDialogRecord]):
        super().__init__(record_list)

    def match(
        self,
        expected_output: List[str],
        real_output: str,
    ) -> bool:
        """
        Use the provided regular expression list to extract output from output, and then judge whether it matches
        :param expected_output: the answer from dataset
        :param real_output: actual output generated by model
        """
        real_output = self.del_chain_of_thought(real_output)
        pred = self.get_answer_from_match_patterns(real_output)
        expected_output = [
            output.split("#### ")[1].replace(",", "") for output in expected_output
        ]
        if not pred:
            return False

        try:
            pred = float(pred) if "." in pred else int(pred)
            for expected in expected_output:
                expected = float(expected) if "." in expected else int(expected)
                if pred == expected or abs(pred - expected) < 1e-6:
                    return True

        except ValueError:
            if pred and pred in expected_output:
                return True

        return False

    def get_answer_from_match_patterns(self, real_output: str):
        """
        Get the answer through comparing match_patterns and output
        """
        from common.uc_eval.utils.prompt_config import match_patterns_gsm8k

        output = real_output.replace("*", "")
        for pattern in match_patterns_gsm8k:
            match = re.search(pattern, output)
            if match:
                clean = re.sub(r"[€£¥$]", "", match.group(1))
                return clean.replace(",", "")

        return None
