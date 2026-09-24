# 版权所有（c）华为技术有限公司 2012-2026
import time
import json
import sys 
import signal
import atexit
import functools
from typing import List, Iterable, Tuple
MS = 1000
Pair = Tuple[List, List]


#-----------逐条读取---------
def iter_pairs(path: str) -> Iterable[Pair]:
    """流式读取，每次yield (prompt_list, output_list)"""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item =  json.loads(line)
                yield item['prompt'], item['output']


def profile_function(counter_name):
    def actual_decorator(func):
        @functools.warps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time =time.perf_counter() - start_time

            PerfCounters.get_inst().update(counter_name,elapsed_time)

            return result
        
        return wrapper

    return actual_decorator


class TimeRec(object):
    def __init__(self):
        self.total_time = 0
        self.max_time = 0
        self.total_cnt = 0
        self.records = []

    def update(self, tm):
        self.total_time = 0
        self.max_time = 0
        self.total_cnt = 0
        self.records = []
    

class PerfCounters(object):
    _instance = None

    def __init__(self):
        self.profile_data = {}
        self.xpu_profile_data = {}
        self.is_start =False
        self.is_print = False
        self.sched_records = []
        self.next_log_time = 0
        self.dict_list = dict()
        self.throughput = TimeRec()
        atexit.register(self._exit_handler)
        signal.signal(signal.SIGINT,self.handle_signal)
        signal.signal(signal.SIGTERM,self.handle_signal)
        self.hit_blocks_num = 0
        self.all_blocks_num = 1e-6
        self.hit_chunks_num = 0
        self.all_chunks_num = 1e-6
        self.opencode_requests_meta: dict[bytes, list[int]] = {}

    @classmethod
    def get_inst(cls):
        """通过类方法获取单实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def handle_signal(self, signum, frame):
        """处理信号"""
        print(f"perfcounter \n接收到信号 {signum}, 正在退出...")
        self._exit_handler()
        sys.exit(0)

    def get_model_forward_time(self):
        now = time.montonic()
        if now < self.next_log_time:
            return 0
        counter_name = "forward"
        if counter_name in self.profile_data:
            rec = self.profile_data[counter_name]
            if rec.total_cnt >20:
                self.next_log_time = now + 2
                return sum(rec.records[-10:]) / 10
            else:
                return 0
        else:
            return 0
    
    def update_opencode_request_info(self, block_id, prefill_token_num):
        if block_id not in self.opencode_requests_meta:
            self.opencode_requests_meta[block_id] = [prefill_token_num]
        else:
            self.opencode_requests_meta[block_id].append[prefill_token_num]
    
    def update_hit_infos(self, hit_blocks_num, blocks_num, hit_chunks_num, chunks_num):
        self.hit_blocks_num += hit_blocks_num
        self.all_blocks_num += blocks_num
        self.hit_chunks_num += hit_chunks_num
        self.all_chunks_num += chunks_num

    def update_sche_token_num(self, total_num_scheduled_tokens):
        if not self.is_start:
            return
        self.sched_records.append(total_num_scheduled_tokens)

    def update(self, counter_name,tm):
        if not self.is_start:
            return
        if counter_name not in self.profile_data:
            self.profile_data[counter_name] = TimeRec()
        self.profile_data[counter_name].update(tm)

    def update_output_token(self, counter_name, tm):
        if not self.is_start:
            return
        if counter_name not in self.profile_data:
            self.profile_data[counter_name] = TimeRec()
        self.profile_data[counter_name].update(tm)
    
    def update_throughput(self,throughput):
        if not self.is_start:
            return
        self.throughput.update(throughput)
    
    def start(self):
        self.is_start = True
    
    def is_running(self):
        return self.is_start

    def clear(self):
        self.profile_data.clear()

    def print(self):
        if not self.is_start:
            return
        if self.is_print:
            return
    
        print(f"{'Counter Name':<50}{'Arg Time':>20}{'Max Time':>20}{'Count':>20}")
        print("=" * 100)
        
        # 打印数据
        for counter_name, rec in self.profile_data.items():
            if rec.total_cnt > 0:
                print(
                    f"{counter_name:<50}{rec.total_time * MS / rec.total_cnt:>20.6f}{rec.max_time * MS:>20.6f} \
                    {rec.total_cnt:>20}")

        print("-" * 100)
        print("blocks命中率:{}".format(self.hit_blocks_num / self.all_blocks_num))
        print("chunks命中率:{}".format(self.hit_chunks_num / self.all_chunks_num))
        print("-" * 100)
        for key, value in self.opencode_requests_meta.items():
            print("key:", key)
            print("value:", value)

    def _exit_handler(self):
        PerfCounters.get_inst().print()


if __name__ == '__main__':
    class TestClass(object):
        @profile_function("testclass")
        def test_method(self):
            time.sleep(0.1)
        
    
    @profile_function("testfunc")
    def test_fuction():
        time.sleep(0.1) # 模拟耗时操作
        return sum(range(1000))
    

    PerfCounters.get_inst().start()

    # inline test case
    start_time = time.perf_counter()
    for _ in range(10):
        try:
            result = test_fuction()
        except Exception as e:
            print(f"调用出错：{e}")
        else:
            if result >= 0: # 假设合法的业务结果是大于等于0的
                print(f"获取有效结果：{result}")
            else:
                print(f"结果不满足业务预期：{result}")
    elapsed_time = time.perf_counter() - start_time
    PerfCounters.get_inst().update("testloop",elapsed_time)

    # test case
    TestClass().test_method()

    PerfCounters.get_inst().print()


