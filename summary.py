import json
from collections import defaultdict


# ================================= all samples ===========================================
def calculate_judge_true_ratio(file_path):
    total = 0
    true_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                total += 1

                if data.get('judge') is True:
                    true_count += 1
            
            except json.JSONDecodeError:
                print(f"警告: 跳过无效的 JSON 行: {line[:50]}...")
                continue

    if total == 0:
        return 0, 0, 0.0
    
    ratio = true_count / total * 100
    return total, true_count, ratio


file_path = "results/Qwen2.5-14B-Instruct_longbench_v2_Single_Document_QA.jsonl"
# total, true_count, ratio = calculate_judge_true_ratio(file_path)

# print(f"总行数: {total}")
# print(f"judge=True 的数量: {true_count}")
# print(f"judge=True 的比例: {ratio:.2f}%")


# ================================= group by length ===========================================

def analyze_judge_by_length(file_path):
    """
    统计JSONL文件中：
    - 总体 judge=True 的比例
    - 按 length (long/medium) 分别统计 judge=True 的比例
    """
    stats = {
        'total': 0,
        'true_count': 0,
        'by_length': defaultdict(lambda: {'total': 0, 'true_count': 0})
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # 统计总体
                stats['total'] += 1
                if data.get('judge') is True:
                    stats['true_count'] += 1
                
                # 按length分类统计
                length = data.get('length')
                if length in ['long', 'medium']:
                    stats['by_length'][length]['total'] += 1
                    if data.get('judge') is True:
                        stats['by_length'][length]['true_count'] += 1
                        
            except json.JSONDecodeError:
                print(f"警告: 第{line_num}行JSON格式无效")
                continue
    
    return stats

def print_results(stats):
    """打印统计结果"""
    print("="*50)
    print("📊 总体统计")
    print("="*50)
    total = stats['total']
    true_count = stats['true_count']
    ratio = (true_count / total * 100) if total > 0 else 0
    print(f"总行数: {total}")
    print(f"judge=True: {true_count}")
    print(f"比例: {ratio:.2f}%")
    
    print("\n" + "="*50)
    print("📊 按 length 分类统计")
    print("="*50)
    
    for length in ['long', 'medium']:
        length_stats = stats['by_length'][length]
        length_total = length_stats['total']
        length_true = length_stats['true_count']
        length_ratio = (length_true / length_total * 100) if length_total > 0 else 0
        
        print(f"\n【{length.upper()}】")
        print(f"  行数: {length_total}")
        print(f"  judge=True: {length_true}")
        print(f"  比例: {length_ratio:.2f}%")


stats = analyze_judge_by_length(file_path)
print_results(stats)
