#!/usr/bin/env python3
import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
import argparse

def analyze_longbench_results(input_file: str, output_file: str):
    """
    analyze LongBench v2 results file
    
    Args:
        input_file: input JSONL file path
        output_file: output Excel file path
    """
    # read data
    samples = []
    seen_ids = set()
    duplicate_ids = []
    
    print(f"reading file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                sample_id = sample.get("_id")
                
                # check duplicates
                if sample_id in seen_ids:
                    duplicate_ids.append((line_num, sample_id))
                else:
                    seen_ids.add(sample_id)
                    samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"warning: line {line_num} JSON parsing failed: {e}")
                continue
    
    print(f"total number of samples: {len(samples)}")
    if duplicate_ids:
        print(f"found {len(duplicate_ids)} duplicate _ids:")
        for line_num, dup_id in duplicate_ids[:10]:  # only show first 10
            print(f"  line {line_num}: {dup_id}")
        if len(duplicate_ids) > 10:
            print(f"  ...and {len(duplicate_ids) - 10} more duplicates")
   
    
 
    domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    # count (domain, sub_domain) combinations
    # use (domain, sub_domain) as key, because the same sub_domain may belong to different domains
    domain_sub_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for sample in samples:
        domain = sample.get("domain", "Unknown")
        sub_domain = sample.get("sub_domain", "Unknown")
        
        # process judge field: it can be a boolean value, string "true"/"false" or other formats
        judge_value = sample.get("judge", False)
        if isinstance(judge_value, bool):
            judge = judge_value
        elif isinstance(judge_value, str):
            judge = judge_value.lower() in ("true", "1", "yes", "correct")
        else:
            judge = bool(judge_value)
        
        # count domain (summary)
        domain_stats[domain]["total"] += 1
        if judge:
            domain_stats[domain]["correct"] += 1
        
        # count (domain, sub_domain) combinations
        key = (domain, sub_domain)
        domain_sub_stats[key]["total"] += 1
        if judge:
            domain_sub_stats[key]["correct"] += 1
    
    # prepare output data - use hierarchical structure: first domain, then sub_domains under the domain
    results = []
    
    # group by domain, then add all sub_domains under the domain
    for domain in sorted(domain_stats.keys()):
        # add domain row (summary) first
        domain_stat = domain_stats[domain]
        results.append({
            "Domain": domain,
            "Sub_Domain": "",  # domain summary row, sub_domain is empty
            "Total_Samples": domain_stat["total"],
            "Correct_Count": domain_stat["correct"],
            "Accuracy": domain_stat["correct"] / domain_stat["total"] if domain_stat["total"] > 0 else 0.0
        })
        
        
        domain_sub_domains = sorted(set([sub for dom, sub in domain_sub_stats.keys() if dom == domain]))
        
        for sub_domain in domain_sub_domains:
            key = (domain, sub_domain)
            sub_stat = domain_sub_stats[key]
            results.append({
                "Domain": domain,
                "Sub_Domain": sub_domain,
                "Total_Samples": sub_stat["total"],
                "Correct_Count": sub_stat["correct"],
                "Accuracy": sub_stat["correct"] / sub_stat["total"] if sub_stat["total"] > 0 else 0.0
            })
    
    # create DataFrame
    df = pd.DataFrame(results)
    
    # save to Excel
    print(f"saving results to: {output_file}")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Statistics', index=False)
        
        # set column width and format
        worksheet = writer.sheets['Statistics']
        worksheet.column_dimensions['A'].width = 40  # Domain
        worksheet.column_dimensions['B'].width = 40    # Sub_Domain
        worksheet.column_dimensions['C'].width = 15    # Total_Samples
        worksheet.column_dimensions['D'].width = 15    # Correct_Count
        worksheet.column_dimensions['E'].width = 15    # Accuracy
        
        # set number format
        from openpyxl.styles import Font
        header_font = Font(bold=True)
        for cell in worksheet[1]:
            cell.font = header_font
        
        # set Accuracy column to percentage format
        from openpyxl.styles.numbers import FORMAT_PERCENTAGE_00
        for row in range(2, len(df) + 2):
            worksheet[f'E{row}'].number_format = FORMAT_PERCENTAGE_00
    
    print(f"analysis completed!")
    print(f"\nstatistics summary:")
    print(f"  - number of domains: {len(domain_stats)}")
    print(f"  - number of (Domain, Sub_Domain) combinations: {len(domain_sub_stats)}")
    print(f"  - total number of samples: {len(samples)}")
    
    # calculate total correct number
    total_correct = 0
    for sample in samples:
        judge_value = sample.get("judge", False)
        if isinstance(judge_value, bool):
            if judge_value:
                total_correct += 1
        elif isinstance(judge_value, str):
            if judge_value.lower() in ("true", "1", "yes", "correct"):
                total_correct += 1
        else:
            if bool(judge_value):
                total_correct += 1
    
    print(f"  - total correct number: {total_correct}")
    print(f"  - overall accuracy: {total_correct / len(samples) * 100:.2f}%")
    
    sub_domain_to_domains = defaultdict(set)
    for domain, sub_domain in domain_sub_stats.keys():
        sub_domain_to_domains[sub_domain].add(domain)
    
    duplicate_subs = {sub: doms for sub, doms in sub_domain_to_domains.items() if len(doms) > 1}
    if duplicate_subs:
        print(f"\nduplicate sub_domain:")
        for sub, doms in sorted(duplicate_subs.items()):
            print(f"  - '{sub}': {sorted(doms)}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="analyze LongBench v2 results")
    parser.add_argument(
        "input_file",
        type=str,
        help="input JSONL file path"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="output Excel file path (optional, default is the same directory as the input file)"
    )
    
    args = parser.parse_args()
    input_file = args.input_file
    
    # if output file is not specified, generate it in the same directory as the input file
    if args.output_file:
        output_file = args.output_file
    else:
        input_path = Path(input_file)
        # generate output file name: replace .jsonl with _analysis.xlsx
        output_file = input_path.parent / f"{input_path.stem}_analysis.xlsx"
    
    # ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    df = analyze_longbench_results(input_file, str(output_file))
    
    # print first 10 rows preview
    print("\nfirst 10 rows preview:")
    print(df.head(10).to_string())