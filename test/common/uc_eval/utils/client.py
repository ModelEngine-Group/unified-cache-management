import os
import time
import json
import requests
import copy
import uuid
import concurrent.futures
from tqdm import tqdm
from typing_extensions import override
from typing import Union, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, PreTrainedTokenizer
from common.uc_eval.utils.utils import get_logger, PathUtil
from common.uc_eval.utils.data_class import (
    RequestRecord,
    MultiTurnDialogRecord,
    ModelConfig,
)

logger = get_logger()
TIMEOUT = 6000
HEADERS = {"User-Agent": "Benchmark Client", "Content-Type": "application/json"}
CHUNK_SIZE = 2**16


def _excute_with_pool(
    task_func: callable,
    process_func: callable,
    tasks: List,
    parallel_num: int,
    desc: str = "Processing Requests",
) -> List[RequestRecord | MultiTurnDialogRecord]:
    record_results: List[RequestRecord | MultiTurnDialogRecord] = []
    if parallel_num > len(tasks):
        logger.error(
            f"The number of requests: {len(tasks)} is less than parallel_num: {parallel_num}, please check..."
        )
        raise ValueError(
            f"The number of requests: {len(tasks)} is less than parallel_num: {parallel_num}, please check..."
        )
    logger.info(f"Start to send {len(tasks)} requests to server...")
    with ThreadPoolExecutor(max_workers=parallel_num) as executor:
        futures = [executor.submit(task_func, task) for task in tasks]

        with tqdm(total=len(futures), desc=desc, mininterval=0.5) as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    pbar.update(1)
                    result = process_func(future.result())
                    record_results.append(result)
                    pbar.set_postfix(
                        {"Completed": len(record_results), "Pending": len(futures) - pbar.n}
                    )
                except Exception as e:
                    pbar.update(1)
                    logger.error(f"Requested failed: {str(e)}")
                    raise Exception(f"Requested failed: {str(e)}")
        return record_results


class BaseClient:
    def __init__(
        self,
        config: ModelConfig,
        stream: bool = False,
        enable_prefix_cache: bool = False,
    ):
        self.ip_ports = config.ip_ports
        self.url = f"http://{self.ip_ports}/v1/chat/completions"
        self.served_model_name = config.served_model_name
        tokenizer_path = PathUtil.get_datasets_dir_path(config.tokenizer_path)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.session = requests.Session()
        self.payload = config.payload
        self.enable_prefix_cache = enable_prefix_cache
        self.stream = stream
        if self.stream:
            self.payload.update({"stream": True, "ignore_eos": True, "temperature": 0.0})
        else:
            self.payload.update({"stream": False, "ignore_eos": False, "temperature": 0.0})

    def handle_requests_with_pool(
        self, prompt_list: List, parallel_num: int, max_tokens: int
    ) -> List[RequestRecord]:
        return _excute_with_pool(
            task_func=lambda prompt: self.send_request(prompt, max_tokens),
            process_func=self.update_request_record,
            tasks=prompt_list,
            parallel_num=parallel_num,
        )

    def send_request(self, prompt, max_tokens) -> List[RequestRecord]:
        """
        update payload and send request
        """
        payload = self._update_payload(prompt, max_tokens)
        if self.stream:
            record = self.do_stream_request(payload)
        else:
            record = self.do_request(payload)
        return record

    def _update_payload(self, prompt, max_tokens) -> Dict:
        """
        update reqeust payload
        """
        payload = copy.deepcopy(self.payload)
        payload.update({"model": self.served_model_name})
        # If payload already has default max_tokens, the input max_tokens will be set to 0
        if max_tokens > 0:
            payload.update({"max_tokens": max_tokens})
        if isinstance(prompt, str):
            message = [{"role": "user", "content": prompt}]
        if isinstance(prompt, list):
            # Multi-turn conversation - prompt already contains full message history.
            # No need to update messages as they are already properly formatted
            message = prompt
        payload.update({"messages": message})

        return payload

    def _create_record(self, prompt):
        # If the prompt is not a dict, it must be a list of dicts for multi-turn dialogue.
        if isinstance(prompt, dict):
            record = RequestRecord(input_data=prompt["content"])
        else:
            record = RequestRecord(input_data=str(prompt))

        return record

    def update_request_record(
        self, records: Union[RequestRecord, List[RequestRecord]]
    ) -> Union[RequestRecord, List[RequestRecord]]:
        """
        Get the number of input and output tokens for each request record
        """
        if not records:
            logger.warning("No records to update, please check...")
        if isinstance(records, RequestRecord):
            single_record = records
            records = [single_record]
        else:
            single_record = None

        for record in records:
            record.input_tokens = len(self.tokenizer.tokenize(record.input_data))
            record.output_tokens = len(self.tokenizer.tokenize(record.output_data))

        return records[0] if single_record is not None else records

    def _requset(self, payload):
        response = None
        try:
            response = self.session.post(
                self.url,
                headers=HEADERS,
                json=payload,
                timeout=TIMEOUT,
                stream=self.stream,
            )
            response.raise_for_status()
            return response
        except Exception as err:
            raise self._handle_request_error(err)

    def do_request(self, payload: Dict) -> RequestRecord:
        prompt = payload["messages"]
        record = self._create_record(prompt)
        record.start_time = time.time()

        response = self._requset(payload)
        result = json.loads(response.text)
        request_id = result.get("id", "request_id not found")
        output = self._get_message_from_response(result)

        record.request_id = request_id
        record.output_data = output
        record.is_success = True
        record.end_time = time.time()
        record.req_cost = record.end_time - record.start_time
        return record

    def _get_message_from_response(self, response) -> str:
        message = response.get("choices", [])[0].get("message", {})
        output = ""
        if message.get("content", "") is not None:
            output += message.get("content", "")
        elif message.get("reasoning_content", "") is not None:
            output += message.get("reasoning_content", "")
        return output

    def do_stream_request(self, payload: Dict) -> RequestRecord:
        prompt = payload["messages"]
        record = self._create_record(prompt)
        while True:
            all_chunks = []
            first_token = True
            last_chunk = None
            timeout_finish_reason = False
            cur_time = last_time = time.perf_counter()
            record.start_time = last_time
            response = self._requset(payload)
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                all_chunks.append(chunk)
                if len(chunk.strip()) == 0:
                    continue
                last_chunk = chunk
                cur_time = time.perf_counter()
                time_diff = cur_time - last_time
                if first_token:
                    record.prefill_latency = time_diff
                    first_token = False
                else:
                    record.tbt_list.append(time_diff)
                last_time = cur_time
                chunk_output = chunk[5:].strip().decode("utf-8")

                # when the MindIE engine side timeout, it will return timeout information
                if chunk.startswith(b"Engine callback timeout"):
                    self._print_request_info(
                        request_id=record.request_id,
                        chunk=chunk,
                        content=record.output_data,
                        all_chunks=all_chunks,
                        payload=payload,
                        msg="Engine callback timeout",
                    )
                    record.output_data = "TIMEOUT"
                    return record
                if "[DONE]" in chunk_output:
                    logger.debug(f"Finished chunk: {chunk_output=}")
                    continue
                output = self._get_message_from_stream_response(json.loads(chunk_output))
                if record.request_id == "":
                    record.request_id = json.loads(chunk_output).get("id", "request_id not found")
                record.output_data += output

                # when the uc-vllm request timeout, finish_reason == "length" and the final output is empty
                finish_reason = (
                    json.loads(chunk_output).get("choices", [])[0].get("finish_reason", "")
                )
                if finish_reason == "length":
                    timeout_finish_reason = True

            # handle the last chunk
            if last_chunk.startswith(b"data:"):
                chunk_output = last_chunk[5:].strip().decode("utf-8")
            else:
                chunk_output = last_chunk.strip().strip().decode("utf-8").rstrip("\0")
            # while the last chunk meets the following conditions, the request is finished successfully
            if "[DONE]" in chunk_output:
                break
            else:
                self._print_request_info(
                    request_id=record.request_id,
                    chunk=chunk,
                    content=record.output_data,
                    all_chunks=all_chunks,
                    payload=payload,
                    msg="request failed, please retry!!!",
                )
                break
        # while the request is done, we need to check the content to see if the request is successful
        if record.output_data == "":
            if timeout_finish_reason:
                self._print_request_info(
                    request_id=record.request_id,
                    chunk=chunk,
                    content=record.output_data,
                    all_chunks=all_chunks,
                    payload=payload,
                    msg="vllm server scheduling timeout, please check",
                )
                return record
            else:
                self._print_request_info(
                    request_id=record.request_id,
                    chunk=chunk,
                    content=record.output_data,
                    all_chunks=all_chunks,
                    payload=payload,
                    msg="the request returned an empty message, which may be an unknown error on the engine side. Please check the specific reason!",
                )
                return record
        record.is_success = True
        record.end_time = time.perf_counter()
        record.req_cost = record.end_time - record.start_time
        logger.debug(f"{record.request_id} finished, cost: {record.req_cost:.2f}s")
        return record

    def _get_message_from_stream_response(self, response) -> str:
        message = response.get("choices", [])[0].get("delta", {})
        output = ""
        if message.get("content", "") is not None:
            output += message.get("content", "")
        elif message.get("reasoning_content", "") is not None:
            output += message.get("reasoning_content", "")
        return output

    def clear_hbm(self) -> bool:
        """
        The API is used to clear HBM. It is available only when the serving backend is VLLM.
        """
        os.environ["NO_PROXY"] = "127.0.0.1, localhost, local, .local"
        logger.info("Begin to clear HBM")
        headers = {"Content-Type": "application/json"}
        payload = {}
        url = f"http://{self.ip_ports}/reset_prefix_cache"
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
        except Exception as err:
            raise self._handle_request_error(err)
        time.sleep(5)
        logger.info("Clear HBM success")
        return True

    def _handle_request_error(self, err: Exception) -> Exception:
        """
        Used to handle request errors
        """
        if isinstance(err, requests.exceptions.ConnectionError):
            logger.error(f"Cannot connect to {self.url}, please check your network")
            return ConnectionError(f"Cannot connect to {self.url}")
        elif isinstance(err, requests.exceptions.Timeout):
            logger.error("The request timed out, please check your server status")
            return TimeoutError("The request timed out, please check your server status")
        elif isinstance(err, requests.exceptions.HTTPError):
            status_code = err.response.status_code
            if status_code == 404:
                logger.error(
                    f"The requested resource does not exist, or the served model name is incorrect"
                )
            else:
                logger.error(f"HTTP error, status code: {status_code}")
            return Exception(f"HTTP error, status code: {status_code}, err: {err}")
        else:
            logger.error(f"Other error: {err}")
            return Exception(f"Other error: {err}")

    @staticmethod
    def _print_request_info(**kwargs):
        """print request info when the request is failed"""
        for key, value in kwargs.items():
            value = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
            logger.error(f"{key} => {value}")


class MultiDialogClient(BaseClient):
    def __init__(self, config: ModelConfig, stream: bool, enable_prefix_cache: bool):
        super().__init__(config, stream, enable_prefix_cache)
        self.uuid = uuid.uuid4().hex

    @override
    def handle_requests_with_pool(
        self, cases: List[List[Union[str, Dict]]], parallel_num: int, max_tokens: int = -1
    ) -> List[List[MultiTurnDialogRecord]]:
        return _excute_with_pool(
            task_func=lambda case: self._send_multi_request(case, max_tokens),
            process_func=self.update_request_record,
            tasks=cases,
            parallel_num=parallel_num,
        )

    def _send_multi_request(
        self, case: List[Union[str, Dict]], max_tokens: int = -1
    ) -> List[MultiTurnDialogRecord]:
        case_name, dialog = case
        history, conv_record = [], []
        conversion = dialog["conversations"]
        turns = self._convert_conversation_2_turns(conversion, 2)
        for i, turn in enumerate(turns):
            in_content, reply = turn[0]["content"], turn[1]["content"]
            # 更新payload，然后发送请求
            prompt = self._update_request_body(history, in_content)
            record: RequestRecord = self.send_request(prompt, max_tokens)
            record.case_name = case_name
            history = self._update_history(history, in_content, reply)
            multi_turn_record: MultiTurnDialogRecord = self._update_multi_turn_request_record(
                record, len(turns), i
            )
            conv_record.append(multi_turn_record)
        return conv_record

    def _update_multi_turn_request_record(
        self, record: RequestRecord, total_turns: int, turn_id: int
    ) -> MultiTurnDialogRecord:
        """
        Update multi-tuen dialogue request record
        """
        request_record = MultiTurnDialogRecord()
        request_record.__dict__.update(record.__dict__)
        request_record.total_turns = total_turns
        request_record.turn_id = turn_id
        return request_record

    @staticmethod
    def _convert_conversation_2_turns(conversion_list: list, chunk_size: int):
        """
        Convert conversation list to turns
        """
        if chunk_size < 0:
            raise ValueError(f"the chunk size {chunk_size} must be greater than 0")
        num_full_chunks = len(conversion_list) // chunk_size
        return [
            conversion_list[i * chunk_size : (i + 1) * chunk_size] for i in range(num_full_chunks)
        ]

    def _update_request_body(self, history: Optional[List[Dict]], in_content: str):
        """
        Multi turn dialogue request body
        """
        history = copy.deepcopy(history)
        if history and self.enable_prefix_cache:
            # To make sure the prefix cache is unique
            history[0]["content"] = f"uuid: [{self.uuid}]" + history[0]["content"]
        if history and not self.enable_prefix_cache:
            history[0]["content"] = f"uuid: [{uuid.uuid4().hex}]" + history[0]["content"]

        message = history + [{"role": "user", "content": in_content}]
        return message

    @staticmethod
    def _update_history(
        history: Optional[List[Dict]], in_content: str, out_content: str
    ) -> List[Dict]:
        """
        Update conversation history
        """
        history.append({"role": "user", "content": in_content})
        history.append({"role": "assistant", "content": out_content})
        return history


class DocQaClient(BaseClient):
    def __init__(self, config: ModelConfig, stream: bool, enable_prefix_cache: bool):
        super().__init__(config, stream, enable_prefix_cache)

    @override
    def handle_requests_with_pool(
        self, cases: List[Union[str, str, str]], parallel_num: int, max_tokens: int = -1
    ) -> List[List[MultiTurnDialogRecord]]:
        return _excute_with_pool(
            task_func=lambda case: self.send_qa_request(case, max_tokens),
            process_func=self.update_request_record,
            tasks=cases,
            parallel_num=parallel_num,
        )

    def send_qa_request(
        self, case: Union[str, str, str, str], max_tokens: int = -1
    ) -> RequestRecord:
        case_name, context, question, answer = case
        prompt = context + question
        record: RequestRecord = self.send_request(prompt, max_tokens)
        record.case_name = case_name
        record.question = question
        record.expected_output = answer
        return record
