import json
import argparse
import tempfile
import os
import shutil

def extract_task_from_response(response):
    try:
        start_idx = response.find('```json\n{')
        if start_idx != -1:
            end_idx = response.find('```', start_idx + 12)
            if end_idx != -1:
                json_str = response[start_idx + 8:end_idx].strip()
                task_data = json.loads(json_str)
                if 'question' in task_data:
                    return task_data['question']
                
        return None
    except:
        return None

def process_file(input_file):
    temp_dir = os.path.dirname(input_file)
    temp_file = os.path.join(temp_dir, '.temp_' + os.path.basename(input_file))
    
    try:
        with open(temp_file, 'w', encoding='utf-8') as temp_out:
            with open(input_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    try:
                        data = json.loads(line.strip())
                        prompt = data.get('prompt', '')
                        responses = data.get('responses', [])
                        problem_id = data.get('id', '')
                        
                        tasks = []
                        for response in responses:
                            task = extract_task_from_response(response)
                            if task:
                                tasks.append(task)
                        
                        if not tasks:
                            tasks = [prompt]
                        
                        for task in tasks:
                            output_data = {"problem_id": problem_id, "question": task}
                            temp_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                        
                    except Exception as e:
                        print(f"Error processing line: {e}")
                        continue
        
        shutil.move(temp_file, input_file)
    except Exception as e:
        print(f"Error processing file: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

def main():
    parser = argparse.ArgumentParser(description='Extract task descriptions from JSONL file')
    parser.add_argument('--input', type=str, default='', help='Input file path')
    
    args = parser.parse_args()
    
    print(f"Starting to process file: {args.input}")
    process_file(args.input)
    print(f"Processing completed, file updated: {args.input}")

if __name__ == "__main__":
    main()