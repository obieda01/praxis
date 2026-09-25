import json, os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=== Starting SWE-Refactor Micro-PoC (5 Tasks) ===")
start_time = time.time()

with open("data/poc_5_tasks.json", "r") as f:
    tasks = json.load(f)

print(f"Successfully loaded {len(tasks)} tasks.")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
print(f"Loading Model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

system_prompt = (
    "You are an expert Java software engineer specialized in behavior-preserving code refactoring. "
    "Perform the requested refactoring accurately while strictly preserving all existing behavior, "
    "syntax correctness, and logic invariants. Do not introduce any new external dependencies."
)

results = []
for idx, task in enumerate(tasks, 1):
    print(f"--- Running Task {idx}/5: [{task['project']}] {task['refactoringType']} ---")
    user_prompt = f"Task: Perform {task['refactoringType']} on Java code.\nJDK: {task['compileJDK']}\n\nCode:\n```java\n{task['sourceCodeBefore']}\n```\nOutput only valid refactored Java code."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=768, temperature=0.2, do_sample=True)
    gen_time = time.time() - t0
    
    gen_tokens = output_ids[0][len(inputs.input_ids[0]):]
    patch = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    print(f"Generated {len(gen_tokens)} tokens in {gen_time:.2f}s ({len(gen_tokens)/gen_time:.1f} tok/s)")
    
    results.append({
        "uniqueId": task["uniqueId"],
        "project": task["project"],
        "refactoringType": task["refactoringType"],
        "compileJDK": task["compileJDK"],
        "generatedPatch": patch,
        "tokensGenerated": len(gen_tokens),
        "inferenceTimeSec": round(gen_time, 2)
    })

os.makedirs("results", exist_ok=True)
with open("results/poc_5_tasks_output.json", "w") as out_f:
    json.dump(results, out_f, indent=2)

print(f"=== Micro-PoC Complete in {time.time() - start_time:.2f}s ===")
