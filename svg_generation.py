from util import load_json_file
from api import call_openai_api
import concurrent.futures
from tqdm import tqdm
import json
import os
import time
import multiprocessing

model_list = ["gpt-4o-mini", "gpt-4o", "o1-mini", "o1", "o3-mini", "o3"]

def get_svg_generation_prompt(description):
    """
    Generate the main SVG generation prompt given a description.
    """
    return (
        "You are an expert in generating SVG code.\n"
        "Your task is to carefully analyze the description and produce only the corresponding SVG code.\n"
        "Do not generate any images or explanations—output strictly the SVG code that fulfills the following description.\n\n"
        "You must generate the SVG code in the following format:\n"
        "1. Start with <svg> tag\n"
        "2. Include all necessary SVG elements and attributes\n"
        "3. End with </svg> tag\n\n"
        "You must ensure that the SVG code is valid and complete.\n\n"
        "You must ensure the SVG code is minimal and only includes elements necessary to satisfy the description.\n\n"
        
        f"Description: {description}"
    )

def parse_svg_from_response(response):
    """
    Extract SVG code from the model response and indicate success.
    
    Returns:
        tuple: (svg_code, success)
            svg_code (str): The extracted SVG code, or empty string if not found.
            success (bool): True if SVG code was successfully extracted, False otherwise.
    """
    svg_start = response.find('<svg')
    svg_end = response.find('</svg>')
    if svg_start == -1 or svg_end == -1:
        return "", False  # Extraction failed
    svg_end += len('</svg>')
    return response[svg_start:svg_end], True  # Extraction succeeded

def generate_single_svg(model, prompt, max_tokens=40000):
    """
    Generate SVG code for a single prompt using the model.
    """
    print(f"Calling API for model: {model}")  # Debug log
    try:
        response = call_openai_api(prompt, model, max_tokens)
        print(f"API call successful for {model}")  # Debug log
        return response
    except Exception as e:
        print(f"Error in generate_single_svg for model {model}: {str(e)}")  # Debug log
        raise

def reprocess_failed_entries(model, result_path, raw_result_path, max_retry=3, max_workers=100, id_key="image_id", prompt_key="caption", bench="coco-val"):
    """
    Read the result JSON file, identify entries where success is false,
    and try to regenerate SVGs for those entries using parallel processing.
    """
    # Load existing results
    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    with open(raw_result_path, "r", encoding="utf-8") as f:
        raw_results = json.load(f)
    
    # Find failed entries
    failed_entries = []
    for i, entry in enumerate(results):
        if not all(entry.get("success", [])):
            failed_entries.append((i, entry))  # Store index along with entry
    
    if not failed_entries:
        print("No failed entries found. All SVGs were generated successfully.")
        return results, raw_results
    
    print(f"Found {len(failed_entries)} failed entries. Attempting to regenerate in parallel...")
    
    # Load original prompts
    if bench == "coco-val":
        benchmark_path = "prompts/coco_test_data.json"
    elif bench == "sgp-val":
        benchmark_path = "prompts/eval.json"
    prompts = load_json_file(benchmark_path)
    id_to_prompt = {entry.get(id_key): entry.get(prompt_key, "") for entry in prompts}
    
    # Define a function to process a single failed entry
    def process_failed_entry(failed_entry_tuple):
        index, failed_entry = failed_entry_tuple
        entry_id = failed_entry[id_key]
        
        if entry_id not in id_to_prompt:
            print(f"Warning: Prompt for ID {entry_id} not found. Skipping.")
            return index, {
                id_key: entry_id,
                "svgs": ["", "", ""],
                "raw_responses": ["", "", ""],
                "success": [False, False, False],
                "caption": ""
            }
        
        entry_prompt = id_to_prompt[entry_id]
        full_prompt = get_svg_generation_prompt(entry_prompt)
        
        svg_results = []
        raw_responses = []
        success_flags = []

        max_tokens = 10000
        if model in ["o3", "o3-mini", "gemini-2.5-pro-preview", "claude-3.7-sonnet:thinking"]:
            max_tokens = 40000
        
        for _ in range(3):  # Generate three times
            for attempt in range(max_retry):
                response = generate_single_svg(model, full_prompt, max_tokens)
                svg, success = parse_svg_from_response(response)
                print(f"Retry ID: {entry_id}, Attempt: {attempt+1}, Success: {success}")
                if success:
                    svg_results.append(svg)
                    raw_responses.append(response)
                    success_flags.append(True)
                    break
            else:  # If all retries failed
                svg_results.append("")
                raw_responses.append(response)
                success_flags.append(False)
        
        return index, {
            id_key: entry_id,
            "svgs": svg_results,
            "raw_responses": raw_responses,
            "success": success_flags,
            "caption": entry_prompt
        }
    
    # Use ThreadPoolExecutor for parallel processing
    updated_entries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(failed_entries))) as executor:
        futures = {executor.submit(process_failed_entry, entry): i for i, entry in enumerate(failed_entries)}
        with tqdm(total=len(failed_entries), desc="Regenerating failed SVGs") as pbar:
            for future in concurrent.futures.as_completed(futures):
                index, result = future.result()
                updated_entries.append((index, result))
                pbar.update(1)
    
    # Update the results and raw_results with the new values
    for index, entry in updated_entries:
        # Update regular results
        results[index] = {
            id_key: entry[id_key],
            "svgs": entry["svgs"],
            "success": entry["success"],
            "caption": entry["caption"]
        }
        
        # Update raw results
        raw_results[index] = {
            id_key: entry[id_key],
            "svgs": entry["svgs"],
            "raw_responses": entry["raw_responses"],
            "success": entry["success"],
            "caption": entry["caption"]
        }
    
    # Save updated results
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(raw_result_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
    
    # Count remaining failures
    remaining_failures = sum(1 for entry in results if not all(entry.get("success", [])))
    print(f"Regeneration complete. Remaining failures: {remaining_failures}")
    
    return results, raw_results

def svg_generation(model, bench):
    """
    随机采样100条数据并批量生成SVG，每个prompt生成三次，并保存原始response。
    在json中也保存是否parse成功(success)。
    如果解析失败，则会再次尝试生成，直到成功为止（最多重试3次）。
    如果仍有失败的条目，将在生成后重新尝试生成这些失败项。
    """
    import random
    import time
    import multiprocessing

    # 根据CPU核心数设置worker数量
    MAX_WORKERS = 200
    max_tokens = 10000
    if model in ["o3", "o3-mini"]:
        max_tokens = 40000
    model = model
    output_dir = "svg_generation_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting SVG generation for model: {model}")
    print(f"Using {MAX_WORKERS} workers for parallel processing")

    # Set up benchmark specific parameters
    if bench == "coco-val":
        benchmark_path = "prompts/coco_test_data.json"
        prompt_key = "caption"
        id_key = "image_id"
    elif bench == "sgp-val":
        benchmark_path = "prompts/eval.json"
        prompt_key = "description"
        id_key = "file_name"
    else:
        raise ValueError(f"Unknown benchmark: {bench}")

    prompts = load_json_file(benchmark_path)
    test_samples = prompts
    print(f"Processing {len(test_samples)} prompts")

    results = []
    raw_results = []
    active_workers = 0
    max_active_workers = 0
    worker_lock = multiprocessing.Lock()

    def process_entry(entry_tuple):
        nonlocal active_workers, max_active_workers
        index, entry = entry_tuple
        entry_id = entry.get(id_key)
        entry_prompt = entry.get(prompt_key, "")
        
        with worker_lock:
            active_workers += 1
            max_active_workers = max(max_active_workers, active_workers)
        
        try:
            full_prompt = get_svg_generation_prompt(entry_prompt)
            
            # Generate three SVGs for each prompt
            svg_results = []
            raw_responses = []
            success_flags = []
            
            for i in range(3):  # Generate three times
                max_retry = 3
                for attempt in range(max_retry):
                    try:
                        response = generate_single_svg(model, full_prompt, max_tokens)
                        svg, success = parse_svg_from_response(response)
                        if success:
                            svg_results.append(svg)
                            raw_responses.append(response)
                            success_flags.append(True)
                            break
                    except Exception as e:
                        if attempt == max_retry - 1:  # Last attempt
                            svg_results.append("")
                            raw_responses.append("")
                            success_flags.append(False)
                else:  # If all retries failed
                    svg_results.append("")
                    raw_responses.append("")
                    success_flags.append(False)
            
            return index, {
                id_key: entry_id,
                "svgs": svg_results,
                "raw_responses": raw_responses,
                "success": success_flags,
                "caption": entry_prompt
            }
        finally:
            with worker_lock:
                active_workers -= 1

    # 使用ThreadPoolExecutor并发生成SVG
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_entry, (i, entry)): i for i, entry in enumerate(test_samples)}
        with tqdm(total=len(test_samples), desc="Generating SVGs") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    index, result = future.result()
                    results.append({
                        id_key: result[id_key],
                        "svgs": result["svgs"],
                        "success": result["success"],
                        "caption": result["caption"]
                    })
                    raw_results.append(result)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing future: {str(e)}")
                    raise
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Maximum number of concurrent workers: {max_active_workers}")

    # 保存结果到JSON文件
    model_name = model.split("/")[-1]
    result_path = os.path.join(output_dir, f"{model_name}_{bench}_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存原始结果到单独的JSON文件
    raw_result_path = os.path.join(output_dir, f"{model_name}_{bench}_raw_result.json")
    with open(raw_result_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
    
    # 检查是否有生成失败的条目，如果有则尝试重新生成
    failed_count = sum(1 for entry in results if not all(entry.get("success", [])))
    if failed_count > 0:
        print(f"Found {failed_count} entries with failed SVG generations. Attempting to regenerate...")
        reprocess_failed_entries(model, result_path, raw_result_path, max_retry=3, id_key=id_key, prompt_key=prompt_key, bench=bench)

if __name__ == "__main__":
    for model in model_list:
        svg_generation(model, "coco-val")
    for model in model_list:
        svg_generation(model, "sgp-val")

    # reprocess_failed_entries("model_name", "svg_generation_results/model_name_coco-val_result.json", "svg_generation_results/model_name_coco-val_raw_result.json", max_retry=3)