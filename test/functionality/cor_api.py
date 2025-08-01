import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
from code_exec_server.code_exec_reqs import exec_test

def correctness_check(id, problem_id, language, file_name, ip):
    if language == "python":
        with open("template_PYTHON.py", "r") as file:
            content = file.read()
    else:
        raise ValueError(f"Unsupported language: {language}")
    
    problem_id = str(problem_id)


    pattern_problem_id = r'data\["id"\] == \d+'
    content = re.sub(pattern_problem_id, f'data["id"] == {id}', content)

    pattern_in_outs = r'json\.load\(open\("/tmp/codeexec/in_outs/\d+\.json"\)\)'
    content = re.sub(pattern_in_outs, f'json.load(open("/tmp/codeexec/in_outs/{problem_id}.json"))', content)

    pattern_in_file_name = r'with open\("/tmp/codeexec/code_wait/\d+\.jsonl", "r"\) as f:'
    content = re.sub(pattern_in_file_name, f'with open("/tmp/codeexec/code_wait/{file_name}.jsonl", "r") as f:', content)

    result = exec_test(ip, content, "")

    return result

if __name__ == "__main__":
    result = correctness_check(1, 245, "python", "code_test", "http://127.0.0.1:8012")
    print(result)
