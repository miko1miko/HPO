import json
import argparse
import multiprocessing
import asyncio
import os
import re
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from security.sec_api import security_check
except ImportError:
    print("Warning: Could not import 'security_check' from 'security.sec_api'.")
    print("Using a MOCK security_check function for demonstration.")

def extract_raw_code_from_markdown(formatted_code_string):
    if not isinstance(formatted_code_string, str):
        return ""
    match = re.search(r'^```[a-zA-Z]*\n(.*?)\n?```$', formatted_code_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    if formatted_code_string.startswith("```") and formatted_code_string.endswith("```"):
        temp_code = formatted_code_string[3:-3]
        if '\n' in temp_code:
            first_line, rest_of_code = temp_code.split('\n', 1)
            if first_line.lower() in ["python", "java", "c", "javascript", "go", "ruby", "php", "cpp"]:
                return rest_of_code.strip()
            else:
                return temp_code.strip()
        else:
             if temp_code.lower() not in ["python", "java", "c", "javascript", "go", "ruby", "php", "cpp"]:
                return temp_code.strip()

    return ""

async def perform_security_analysis_async(code_content, language):
    if not code_content or not code_content.strip():
        return [], [""]

    is_insecure, _ , issues_objects = await security_check(code_content, language)
    
    cwe_list = []
    processed_issues_str_list = []

    if is_insecure and issues_objects:
        for issue in issues_objects:
            cwe_id_value = None
            try:
                if hasattr(issue, 'cwe_id'):
                    cwe_id_value = issue.cwe_id
                elif isinstance(issue, dict) and 'cwe_id' in issue:
                    cwe_id_value = issue['cwe_id']
                
                if cwe_id_value:
                    cwe_id_str = str(cwe_id_value).strip()
                    if cwe_id_str and cwe_id_str not in cwe_list:
                        cwe_list.append(cwe_id_str)
            except Exception as e:
                print(f"Warning: Could not extract CWE ID from an issue object ('{str(issue)[:50]}...'): {e}")
            
            processed_issues_str_list.append(str(issue))

    if not processed_issues_str_list:
        processed_issues_str_list = [""]

    return cwe_list, processed_issues_str_list

def worker_analyze_item(args_tuple):
    item, language = args_tuple
    
    formatted_code = item.get('code', '')
    raw_code = extract_raw_code_from_markdown(formatted_code)

    item_id_for_log = item.get('id', item.get('problem_id', 'Unknown_ID'))

    try:
        cwe_results, issues_results = asyncio.run(perform_security_analysis_async(raw_code, language))
    except Exception as e:
        print(f"Error during security analysis for item {item_id_for_log}: {e}")
        cwe_results = ["ANALYSIS_FAILED"]
        issues_results = [f"Error during analysis: {str(e)}"]

    item['cwe'] = cwe_results
    item['issues'] = issues_results
    return item

async def statistic_sec(input_file):
    total_codes = 0
    total_vulnerable = 0
    analysis_failures = 0
    cwe_counts = {}
    
    print(f"\nCalculating statistics from: {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    total_codes += 1
                    
                    cwe_list_from_data = data.get('cwe', [])
                    
                    if "ANALYSIS_FAILED" in cwe_list_from_data or "ERROR_ANALYSIS" in cwe_list_from_data :
                        analysis_failures +=1
                    
                    actual_cwes = [cwe for cwe in cwe_list_from_data if cwe and cwe not in ["ANALYSIS_FAILED", "ERROR_ANALYSIS"]]
                    
                    has_vulnerabilities = bool(actual_cwes)
                    
                    if has_vulnerabilities:
                        total_vulnerable += 1
                        for cwe_id in actual_cwes:
                            cwe_counts[cwe_id] = cwe_counts.get(cwe_id, 0) + 1
                except json.JSONDecodeError:
                    print(f"  Skipping malformed JSON line during statistics calculation: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: Statistics input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during statistics calculation: {e}")
        return None

    vulnerability_rate = total_vulnerable / total_codes if total_codes > 0 else 0
    
    cwe_rates = {cwe: count / total_codes for cwe, count in cwe_counts.items() if total_codes > 0}
    
    print("\n=== Security Statistics ===")
    print(f"Total items processed: {total_codes}")
    if analysis_failures > 0:
        print(f"Items with analysis failures: {analysis_failures} ({analysis_failures/total_codes:.2%})")
    print(f"\n== Vulnerability Statistics ==")
    print(f"Total vulnerable items (with at least one CWE): {total_vulnerable}")
    print(f"Vulnerability rate (based on all processed items): {vulnerability_rate:.2%}")
    
    print("\n== CWE Distribution (count and rate based on all processed items) ==")
    if cwe_counts:
        sorted_cwes = sorted(cwe_counts.items(), key=lambda item: item[0]) 
        for cwe, count in sorted_cwes:
            rate = cwe_rates.get(cwe, 0)
            print(f"{cwe}: Count={count}, Rate={rate:.2%}")
    else:
        print("No specific CWEs found or counted.")
    
    return {
        "total_codes": total_codes,
        "total_vulnerable": total_vulnerable,
        "analysis_failures": analysis_failures,
        "vulnerability_rate": vulnerability_rate,
        "cwe_counts": cwe_counts,
        "cwe_rates": cwe_rates
    }

def main():
    parser = argparse.ArgumentParser(description="Perform multi-threaded security checks on a code dataset.")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the input JSONL file (e.g., code.jsonl from previous script).")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Path for the output JSONL file with added security results.")
    parser.add_argument("--language", type=str, default="python", 
                        help="Programming language of the code (to be passed to security_check).")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(), 
                        help="Number of worker processes. Defaults to CPU count.")
    parser.add_argument("--skip_stats", action="store_true",
                        help="Skip the final statistics calculation step.")
    args = parser.parse_args()

    print(f"Starting security analysis with the following configuration:")
    print(f"  Input File: {args.input_file}")
    print(f"  Output File: {args.output_file}")
    print(f"  Language: {args.language}")
    print(f"  Number of Workers: {args.num_workers}")

    tasks_to_process = []
    print(f"\nReading input file: {args.input_file}...")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                try:
                    tasks_to_process.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"  Skipping invalid JSON on line {i+1}: {line.strip()} - Error: {e}")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found. Exiting.")
        return
    except Exception as e:
        print(f"Error reading input file '{args.input_file}': {e}. Exiting.")
        return

    if not tasks_to_process:
        print("No tasks found in the input file. Exiting.")
        return
    
    print(f"Successfully loaded {len(tasks_to_process)} items for analysis.")

    worker_function_args = [(task_item, args.language) for task_item in tasks_to_process]

    processed_results = []
    
    print(f"\nInitializing multiprocessing pool with {args.num_workers} workers...")
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        print("Processing items...")
        for result_item in tqdm(pool.imap_unordered(worker_analyze_item, worker_function_args), 
                                total=len(tasks_to_process), desc="Analyzing Code Snippets"):
            processed_results.append(result_item)
    
    print(f"\nWriting {len(processed_results)} processed items to: {args.output_file}...")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for item in processed_results:
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("Output successfully saved.")
    except IOError as e:
        print(f"Error writing to output file '{args.output_file}': {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file writing: {e}")
        return

    print("\nSecurity analysis complete!")
    print(f"Results are available in: {args.output_file}")

    if not args.skip_stats:
        asyncio.run(statistic_sec(args.output_file))
    else:
        print("\nSkipping statistics calculation as per --skip_stats flag.")

if __name__ == "__main__":
    main()