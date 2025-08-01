import json
import sys
import asyncio
import os
import re
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from cor_api import correctness_check

def run_correctness_check_sync(id, problem_id, language, file_name, ip):
    return correctness_check(id, problem_id, language, file_name, ip)

async def cor_test(id, problem_id, language, file_name, ip):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, run_correctness_check_sync, id, problem_id, language, file_name, ip
        )

    result_str = result[1].rsplit('\n', 1)[0]
    
    if not result[0]:
        return [-1], result_str
    
    try:
        matches = re.findall(r'\[(.*?)\]', result[1].rsplit('\n', 1)[0])
        if matches:
            match = matches[-1]
            if match:
                content = match
                test_results = []
                for x in content.split(','):
                    x = x.strip()
                    if x == '-1':
                        test_results.append(-1)
                    else:
                        test_results.append(x == 'True')
                return test_results, result_str
    except Exception:
        pass
    
    return [-1], result_str

async def process_line_task(line, language, file_name, ip, semaphore):
    async with semaphore:
        try:
            data = json.loads(line.strip())
            question = data.get('question', '')
            id = data.get('id', '')
            problem_id = data.get('problem_id', '')
            problem_id = str(problem_id)
            code = data.get('code', '')
            start_fence = f"```{language}"
            end_fence = "```"

            cleaned_code = code.strip()

            if cleaned_code.startswith(start_fence):
                cleaned_code = cleaned_code[len(start_fence):]

            if cleaned_code.endswith(end_fence):
                cleaned_code = cleaned_code[:-len(end_fence)]

            code = cleaned_code.strip()
            
            test_list, result_str = await cor_test(id, problem_id, language, file_name, ip)
            
            pass_rate = 0.0
            if test_list and -1 not in test_list:
                total_tests = len(test_list)
                if total_tests > 0:
                    passed_tests = sum(1 for x in test_list if x is True)
                    pass_rate = passed_tests / total_tests
                else:
                    pass_rate = 0.0

            new_data = {
                "id": id,
                "problem_id": problem_id,
                "code": code,
                "question": question,
                "test": test_list,
                "test_result": result_str,
                "pass_rate": pass_rate
            }
            
            return json.dumps(new_data)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Error processing line for id {data.get('id', 'N/A')}: {e}", file=sys.stderr)
            return None

async def process_file(input_file, output_file, language, file_name, ip, threads):
    semaphore = asyncio.Semaphore(threads)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    tasks = [process_line_task(line, language, file_name, ip, semaphore) for line in lines]
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing (correctness mode)"):
            result = await future
            if result:
                outfile.write(result + '\n')


import numpy as np
import itertools

def estimate_pass_at_k(num_samples, num_correct, k):
    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1, dtype=np.float64))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


async def statistic_cor(output_file):
    total_codes = 0
    total_executable = 0
    total_correct = 0
    total_partial = 0
    total_failed = 0
    total_test_cases = 0
    total_passed_test_cases = 0

    problem_stats = defaultdict(list)

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            problem_id = data.get('problem_id')
            total_codes += 1

            test_list = data.get('test', [])
            is_executable = test_list and -1 not in test_list
            if problem_id:
                is_correct = is_executable and all(test_list)
                problem_stats[problem_id].append(is_correct)

            if is_executable:
                total_executable += 1
                all_passed = all(test_list)
                some_passed = any(test_list)
                
                total_test_cases += len(test_list)
                total_passed_test_cases += sum(1 for x in test_list if x)
                
                if all_passed:
                    total_correct += 1
                elif some_passed:
                    total_partial += 1
                else:
                    total_failed += 1

    print("\n=== Statistics ===")
    print(f"Total codes processed: {total_codes}")

    executable_rate = total_executable / total_codes if total_codes > 0 else 0
    correct_rate = total_correct / total_executable if total_executable > 0 else 0
    partial_rate = total_partial / total_executable if total_executable > 0 else 0
    failed_rate = total_failed / total_executable if total_executable > 0 else 0
    test_case_pass_rate = total_passed_test_cases / total_test_cases if total_test_cases > 0 else 0
    count = 0

    if not problem_stats:
        print("\nNo problems found to calculate pass@1")
    else:
        num_samples_per_problem = []
        num_correct_per_problem = []
        for problem_id, results in problem_stats.items():
            n = len(results)
            c = sum(results)
            num_samples_per_problem.append(n)
            num_correct_per_problem.append(c)
            print(f"问题 {problem_id}: {c}/{n} 正确")
            count += n
        pass_at_1_estimates = estimate_pass_at_k(num_samples_per_problem, num_correct_per_problem, 1)
        
        overall_pass_at_1 = pass_at_1_estimates.mean()
        print(f"\n==Correctness Statistics==")
        print(f"Total unique problems: {len(problem_stats)}")
        print(f"pass@1 Rate: {overall_pass_at_1:.2%}")
        print(count)
    
    print(f"Executable codes: {total_executable}")
    print(f"Executable rate: {executable_rate:.2%}")
    print(f"Fully correct rate: {correct_rate:.2%}")
    print(f"Partially correct rate: {partial_rate:.2%}")
    print(f"Failed rate: {failed_rate:.2%}")
    print(f"\n==Test Case Statistics==")
    print(f"Total test cases: {total_test_cases}")
    print(f"Total passed test cases: {total_passed_test_cases}")
    print(f"Test case pass rate: {test_case_pass_rate:.2%}")

async def main():
    parser = argparse.ArgumentParser(description="Process and analyze correctness code")
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--language", type=str, default="python", help="Programming language")
    parser.add_argument("--file_name", type=str, required=True, help="File name for analysis")
    parser.add_argument("--ip", type=str, default="http://127.0.0.1:8000", help="IP address for correctness check API")
    parser.add_argument("--threads", type=int, default=10, help="Number of concurrent tasks to run")
    
    args = parser.parse_args()
    print(args)

    if os.path.exists(args.output):
        print(f"Output file {args.output} already exists. Performing statistics...")
        await statistic_cor(args.output)
    else:
        print(f"Output file {args.output} does not exist. Processing input file...")
        await process_file(args.input, args.output, args.language, args.file_name, args.ip, args.threads)
        print(f"\nProcessing complete. Output written to {args.output}")
        await statistic_cor(args.output)

if __name__ == "__main__":
    asyncio.run(main())