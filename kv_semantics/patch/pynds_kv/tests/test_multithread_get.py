import argparse
import gc
import mmap
import numpy as np
import pynds_kv
import time
import threading
from collections import defaultdict
import logging

parser = argparse.ArgumentParser(description="KV多线程性能测试")
parser.add_argument("--num_blocks", type=int, default=128, help="每批次block数量")
parser.add_argument("--token_size", type=int, default=128 * 1024, help="单个token大小(字节), 默认128KB")
parser.add_argument("--num_tokens_per_block", type=int, default=256, help="每个block中token数量, 默认256 (128KB*256=32MB)")
parser.add_argument("--put_threads", type=int, default=8, help="PUT线程数")
parser.add_argument("--get_threads", type=int, default=8, help="GET线程数")
parser.add_argument("--duration", type=int, default=30, help="测试时长(秒)")
parser.add_argument("--mode", type=str, default="w", choices=["w", "r", "rw"], help="读写模式: w只写, r只读, rw混合")
parser.add_argument("--monitor_interval", type=float, default=1, help="监控时间窗口(秒), 大数据建议增大以减少误差")
parser.add_argument("--nds_config_path", type=str, default=None, help="NDS配置文件路径(nds_config.conf)")
parser.add_argument("--nsids", type=int, nargs='+', default=[3618096867], help="NSID列表，多个时按线程轮询分配")
args = parser.parse_args()

NUM_BLOCKS = args.num_blocks
TOKEN_SIZE = args.token_size
NUM_TOKENS_PER_BLOCK = args.num_tokens_per_block
ONE_LAYER_CACHE_SIZE = TOKEN_SIZE * NUM_TOKENS_PER_BLOCK
TOTAL_SIZE = NUM_BLOCKS * ONE_LAYER_CACHE_SIZE

BATCH_PUT_THREAD_NUM = args.put_threads
BATCH_GET_THREAD_NUM = args.get_threads

TEST_DURATION = args.duration
MONITOR_INTERVAL = args.monitor_interval

READ_WRITE_MODE = args.mode

GB = 1024**3
NSIDS = args.nsids
NDS_CONFIG_PATH = args.nds_config_path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

stop_event = threading.Event()

pre_loaded_block_ids = []

class ThreadStats:
    def __init__(self):
        self.io_count = 0
        self.zero_io_count = 0
        self.total_bytes = 0
        self.total_latency = 0
        self.success_ops = 0
        self.error_count = 0

        self.all_io_count = 0
        self.all_zero_io_count = 0
        self.all_total_bytes = 0
        self.all_total_latency = 0
        self.all_success_ops = 0
        self.all_error_count = 0

    def update(self, ret_code, bytes_transferred, latency=0):
        self.io_count += 1
        self.all_io_count += 1
        if ret_code == 0:
            self.total_bytes += bytes_transferred
            self.all_total_bytes += bytes_transferred

        if latency > 0:
            self.total_latency += latency
            self.all_total_latency += latency

        if ret_code == 0:
            self.success_ops += 1
            self.all_success_ops += 1
        else:
            self.error_count += 1
            self.all_error_count += 1

    def reset(self):
        self.io_count = 0
        self.zero_io_count = 0
        self.total_bytes = 0
        self.total_latency = 0
        self.success_ops = 0
        self.error_count = 0

class GlobalStats:
    def __init__(self):
        self.start_time = time.time()
        self.bw_start_time = time.time()
        self.thread_stats = []
        self.total_block_ids_processed = 0
        self.all_time = 0
        self.max_bandwidth_gbs = 0

        for i in range(BATCH_PUT_THREAD_NUM + BATCH_GET_THREAD_NUM):
            self.thread_stats.append(ThreadStats())

    def add_batch_stats(self, thread_idx, total_size, latency, success):
        bytes_transferred = total_size
        ret_code = 0 if success else -1
        self.thread_stats[thread_idx].update(ret_code, bytes_transferred, latency)

    def get_total_ops(self):
        result = sum(stats.io_count for stats in self.thread_stats)
        return result

    def reset_total_ops(self):
        for stats in self.thread_stats:
            stats.io_count = 0

    def get_all_total_ops(self):
        result = sum(stats.all_io_count for stats in self.thread_stats)
        return result

    def get_total_bytes(self):
        result = sum(stats.total_bytes for stats in self.thread_stats)
        for stats in self.thread_stats:
            stats.total_bytes = 0
        return result

    def reset_total_bytes(self):
        for stats in self.thread_stats:
            stats.total_bytes = 0

    def get_all_total_bytes(self):
        result = sum(stats.all_total_bytes for stats in self.thread_stats)
        return result

    def get_total_errors(self):
        result = sum(stats.error_count for stats in self.thread_stats)
        return result

    def reset_total_errors(self):
        for stats in self.thread_stats:
            stats.error_count = 0

    def get_all_total_errors(self):
        result = sum(stats.all_error_count for stats in self.thread_stats)
        return result

    def get_avg_latency(self):
        total_ops = self.get_total_ops()
        if total_ops == 0:
            return 0

        total_latency = sum(stats.total_latency for stats in self.thread_stats)
        for stats in self.thread_stats:
            stats.total_latency = 0
        return total_latency / total_ops

    def get_all_avg_latency(self):
        total_ops = sum(stats.all_io_count for stats in self.thread_stats)
        if total_ops == 0:
            return 0
        total_latency = sum(stats.all_total_latency for stats in self.thread_stats)
        return total_latency / total_ops

    def get_bandwidth_gbs(self):
        current_time = time.time()
        elapsed_time = current_time - self.bw_start_time
        self.all_time += elapsed_time
        bandwidth = 0
        if elapsed_time == 0:
            bandwidth = 0
        else:
            total_bytes = self.get_total_bytes()
            self.reset_total_bytes()
            bandwidth = total_bytes / elapsed_time / GB
        self.bw_start_time = current_time
        if bandwidth > self.max_bandwidth_gbs:
            self.max_bandwidth_gbs = bandwidth
        return bandwidth

    def get_all_bandwidth_gbs(self):
        elapsed_time = self.all_time
        bandwidth = 0
        if elapsed_time == 0:
            bandwidth = 0
        else:
            total_bytes = sum(stats.all_total_bytes for stats in self.thread_stats)
            bandwidth = total_bytes / elapsed_time / GB
        return bandwidth

current_block_index = 0
pre_loaded_block_ids = []

thread_blocks = {}

def load_block_ids_from_file():
    global pre_loaded_block_ids
    print("    正在从key.txt加载blockID...")
    try:
        with open("key.txt", "r") as f:
            lines = f.readlines()
            if not lines:
                raise ValueError("key.txt文件为空")
            pre_loaded_block_ids = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        block_id = int(line)
                        pre_loaded_block_ids.append(block_id)
                    except ValueError:
                        print(f"    警告: 跳过无效行：{line}")
                        continue
            if not pre_loaded_block_ids:
                raise ValueError("未找到有效的blockID")
            print(f"    成功加载 {len(pre_loaded_block_ids)} 个blockID")
    except FileNotFoundError:
        print("    警告: 未找到key.txt文件")
    except Exception as e:
        print(f"    加载blockids失败：{e}")
    if len(pre_loaded_block_ids) < NUM_BLOCKS:
        pre_loaded_block_ids = list(range(NUM_BLOCKS * 1024))
        print(f"    已生成 {len(pre_loaded_block_ids)} 个顺序blockID")

def get_next_block_ids(current_block_index: int, count: int) -> list:
    start_index = current_block_index
    end_index = start_index + count
    block_ids = []
    if end_index <= len(pre_loaded_block_ids):
        block_ids = pre_loaded_block_ids[start_index:start_index + count]
        current_block_index +=  count
    else:
        remaining = end_index - len(pre_loaded_block_ids)
        block_ids = pre_loaded_block_ids[start_index:len(pre_loaded_block_ids)]
        block_ids.extend(pre_loaded_block_ids[:remaining])
        current_block_index = end_index % len(pre_loaded_block_ids)
    return block_ids, current_block_index

def batch_put_phase(thread_idx: int):
    global thread_blocks, global_stats
    current_block_index = 0
    while not stop_event.is_set():
        try:
            block_ids, current_block_index = get_next_block_ids(current_block_index, NUM_BLOCKS)

            offsets = [0] * NUM_BLOCKS
            lengths = [ONE_LAYER_CACHE_SIZE] * NUM_BLOCKS
            batch_blocks = thread_blocks[thread_idx][:NUM_BLOCKS]
            nsid = NSIDS[thread_idx % len(NSIDS)]
            nsids = [nsid] * NUM_BLOCKS
            start_time = time.time()
            ret = pynds_kv.batchPut(block_ids, batch_blocks, offsets, lengths, nsids)
            end_time = time.time()
            latency  = end_time - start_time
            success = (ret == 0)
            global_stats.add_batch_stats(thread_idx, TOTAL_SIZE, latency, success)
        except Exception as e:
            print(f"    线程{thread_idx} batchPut异常：{e}")
            stop_event.set()

def batch_get_phase(thread_idx: int):
    global thread_blocks, global_stats
    current_block_index = 0
    while not stop_event.is_set():
        try:
            block_ids, current_block_index = get_next_block_ids(current_block_index, NUM_BLOCKS)

            offsets = [0] * NUM_BLOCKS
            lengths = [ONE_LAYER_CACHE_SIZE] * NUM_BLOCKS
            batch_blocks = thread_blocks[thread_idx][:NUM_BLOCKS]
            nsid = NSIDS[thread_idx % len(NSIDS)]
            nsids = [nsid] * NUM_BLOCKS
            start_time = time.time()
            ret = pynds_kv.batchGet(block_ids, batch_blocks, offsets, lengths, nsids)
            end_time = time.time()
            latency  = end_time - start_time
            success = (ret == 0)
            global_stats.add_batch_stats(thread_idx, TOTAL_SIZE, latency, success)
        except Exception as e:
            print(f"    线程{thread_idx} batchPut异常：{e}")
            stop_event.set()

def monitor_thread():
    global global_stats
    print(f"    监控线程启动，将每秒报告性能指标...")
    while not stop_event.is_set():
        try:
            time.sleep(MONITOR_INTERVAL)
            if stop_event.is_set():
                break
            elapsed_time = time.time() - global_stats.start_time
            bandwidth_gbs = global_stats.get_bandwidth_gbs()
            avg_latency_s = global_stats.get_avg_latency()
            total_errors = global_stats.get_total_errors()
            total_ops = global_stats.get_total_ops()
            global_stats.reset_total_ops()
            zero_io_ratio = 0
            if total_ops == 0:
                zero_io_ratio = 0
            else:
                zero_io_ratio = total_errors / total_ops
            print(f"[监控] {elapsed_time:6.1f}s - 带宽: {bandwidth_gbs:3.2f} GB/s, " f"平均延迟: {avg_latency_s:2.6f}s, " f"错误: {total_errors:3d}, " f"零IO率: {zero_io_ratio:5.1f}%")
        except KeyboardInterrupt:
            print(f"    监控线程收到中断信号...")
            stop_event.set()
            break
        except Exception as e:
            print(f"    监控线程异常：{e}")
            stop_event.set()
            break

def worker_thread(thread_idx: int):
    try:
        if READ_WRITE_MODE == "rw":
            import random
            operation = "put" if random.random() < 0.7 else "get"
            if operation == "put":
                batch_put_phase(thread_idx)
            else:
                batch_get_phase(thread_idx)
        elif READ_WRITE_MODE == "w":
            batch_put_phase(thread_idx)
        elif READ_WRITE_MODE == "r":
            batch_get_phase(thread_idx)
    except Exception as e:
            print(f"    线程{thread_idx}异常：{e}")
            stop_event.set()

def print_final_report():
    global global_stats
    print("\n" + "=" * 80)
    print("性能测试完成 - 最终统计报告".center(80))
    print("=" * 80)
    elapsed_time = time.time() - global_stats.start_time
    print(f"测试时长:           {elapsed_time:9.2f} 秒")
    print(f"测试模式:           {'批处理'} ({NUM_BLOCKS}/批)")
    print(f"运行方式:           {READ_WRITE_MODE}")
    print(f"TOKEN_SIZE:           {TOKEN_SIZE} 字节 ({TOKEN_SIZE / 1024:.0f} KB)")
    print(f"NUM_TOKENS_PER_BLOCK:           {NUM_TOKENS_PER_BLOCK}")
    print(f"ONE_LAYER_CACHE_SIZE:           {ONE_LAYER_CACHE_SIZE} 字节 ({ONE_LAYER_CACHE_SIZE / 1024 / 1024:.2f} MB)")
    print(f"使用预定义keys:           {len(pre_loaded_block_ids)} 个")
    print("-" * 80)

    total_bytes = global_stats.get_all_total_bytes()
    total_errors = global_stats.get_all_total_errors()
    bandwidth_gbs = global_stats.get_all_bandwidth_gbs()
    avg_latency_s = global_stats.get_all_avg_latency()
    total_ops = global_stats.get_all_total_ops()
    print(f"总传输字节数:           {total_bytes:12d} B ({total_bytes/GB:6.2f}) GB")
    print(f"总错误次数:           {total_errors:12d}")
    print(f"错误率:           {total_errors/max(total_ops, 1) * 100:12.3f}%")
    print("-" * 80)
    print(f"平均带宽:           {bandwidth_gbs:5.2f} GB/s")
    print(f"最高带宽:           {global_stats.max_bandwidth_gbs:5.2f} GB/s")
    print(f"平均延迟:           {avg_latency_s:3.6f} s")
    print("-" * 80)

def run_performance_test():
    global stop_event, global_stats, buf, blocks, thread_blocks
    print("=" * 80)
    print("kv示例多线程性能测试".center(80))
    print("=" * 80)
    print(f"NUM_BLOCKS:        {NUM_BLOCKS}")
    print(f"TOKEN_SIZE:        {TOKEN_SIZE} 字节 ({TOKEN_SIZE / 1024:.0f} KB)")
    print(f"NUM_TOKENS_PER_BLOCK:        {NUM_TOKENS_PER_BLOCK}")
    print(f"ONE_LAYER_CACHE_SIZE:        {ONE_LAYER_CACHE_SIZE} 字节 ({ONE_LAYER_CACHE_SIZE / 1024 / 1024:.2f} MB) = TOKEN_SIZE * NUM_TOKENS_PER_BLOCK")
    print(f"TOTAL_SIZE:        {TOTAL_SIZE} 字节 ({TOTAL_SIZE / 1024 / 1024:.2f} MB)")
    print(f"TEST_DURATION:        {TEST_DURATION}")
    print(f"PUT线程数:        {BATCH_PUT_THREAD_NUM}")
    print(f"GET线程数:        {BATCH_GET_THREAD_NUM}")
    print(f"读写模式:        {READ_WRITE_MODE}")
    print("=" * 80)

    if READ_WRITE_MODE == "w":
        print(" 操作模式: 只执行写操作 (batchPut)")
    elif READ_WRITE_MODE == "r":
        print(" 操作模式: 只执行读操作 (batchGet)")
    elif READ_WRITE_MODE == "rw":
        print(" 操作模式: 读写混合 (70% 写, 30%读)")
    print("=" * 80)
    stop_event.clear()
    cache_results = {}
    total_errors = 0
    load_block_ids_from_file()
    print("=" * 80)
    total_threads  = BATCH_PUT_THREAD_NUM if READ_WRITE_MODE == "w" else 0
    total_threads  = BATCH_GET_THREAD_NUM if READ_WRITE_MODE == "r" else total_threads
    total_threads  = BATCH_PUT_THREAD_NUM + BATCH_GET_THREAD_NUM if READ_WRITE_MODE == "rw" else total_threads

    print(f"NSID列表:        {NSIDS} (共 {len(NSIDS)} 个)")
    for thread_idx in range(total_threads):
        print(f"    线程{thread_idx:3d} -> nsid {NSIDS[thread_idx % len(NSIDS)]}")

    print(f">>> Phase I: 分配 {TOTAL_SIZE * total_threads} 字节的页面对齐缓冲区")
    mm = mmap.mmap(-1, TOTAL_SIZE * total_threads, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    buf = np.frombuffer(mm, dtype=np.uint8, count = TOTAL_SIZE * total_threads)
    addr = buf.__array_interface__["data"][0]
    print(f"    缓冲区地址 = 0x{addr:x}")
    print(f"    4096字节对齐? {'YES' if addr % 4096 == 0 else 'NO'}")

    print(f">>> Phase II: 调用pynds_kv.init()")
    try:
        ret = pynds_kv.init(buf, TOTAL_SIZE * total_threads, NDS_CONFIG_PATH)
        print(f"    init() 返回值: = {ret}")
        if ret != 0:
            print("!!! 初始化失败")
            mm.close()
            return
    except KeyboardInterrupt:
        print("    收到键盘中断, 停止测试...")
        stop_event.set()

    print(f">>> Phase III: 预生成测试blocks (零拷贝视图, 跳过数据填充)...")
    thread_blocks = {}
    print(f"    总线程数: = {total_threads}")
    for thread_idx in range(total_threads):
        thread_blocks[thread_idx] = []
        for i in range(NUM_BLOCKS):
            offset = thread_idx * NUM_BLOCKS * ONE_LAYER_CACHE_SIZE + i * ONE_LAYER_CACHE_SIZE
            if offset + ONE_LAYER_CACHE_SIZE > TOTAL_SIZE * total_threads:
                print(f"    警告: 线程{thread_idx}的blocks超出缓冲区范围, 减少block数量")
                break
            block = buf[offset:offset + ONE_LAYER_CACHE_SIZE]
            # 零拷贝: 直接使用mmap零初始化的缓冲区视图, 不填充数据
            thread_blocks[thread_idx].append(block)
    print(f"    block预生成完成, 共{total_threads}个线程的blocks")

    print(f">>> Phase IV: 启动监控和测试线程")
    monitor_th = None
    worker_threads = []

    try:
        global_stats = GlobalStats()
        monitor_th = threading.Thread(target=monitor_thread, daemon=True)
        monitor_th.start()

        thread_count = 0
        if READ_WRITE_MODE == "w" or READ_WRITE_MODE == "rw":
            print(f"   启动 {BATCH_PUT_THREAD_NUM} 个PUT线程...")
            for i in range(BATCH_PUT_THREAD_NUM):
                thread = threading.Thread(target=worker_thread, args=(thread_count,))
                thread.daemon = True
                thread.start()
                worker_threads.append(thread)
                thread_count += 1

        if READ_WRITE_MODE == "r" or READ_WRITE_MODE == "rw":
            print(f"   启动 {BATCH_GET_THREAD_NUM} 个GET线程...")
            for i in range(BATCH_GET_THREAD_NUM):
                thread = threading.Thread(target=worker_thread, args=(thread_count,))
                thread.daemon = True
                thread.start()
                worker_threads.append(thread)
                thread_count += 1
        print(f"    共启动 {thread_count} 个工作线程 + 1 个监控线程")

        start_time = time.time()
        while time.time() - start_time <  TEST_DURATION:
            if stop_event.is_set():
                print(f"    检测到异常, 提前退出测试")
                break
            time.sleep(0.1)
        print("    测试时间达到, 停止所有线程...")
        stop_event.set()
    except KeyboardInterrupt:
        print("    收到键盘中断, 停止测试...")
        stop_event.set()
    except Exception as e:
        print(f"    测试异常：{e}")
        stop_event.set()
    finally:
        if monitor_th and monitor_th.is_alive():
            monitor_th.join(timeout=2)
        for thread in worker_threads:
            if thread.is_alive():
                thread.join(timeout=2)
        print_final_report()

        del thread_blocks
        del buf
        del global_stats
        print(">>>测试完成")

if __name__ == "__main__":
    try:
        run_performance_test()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序异常：{e}")
    finally:
        try:
            if 'stop_event' in globals():
                stop_event.set()
        except:
            pass