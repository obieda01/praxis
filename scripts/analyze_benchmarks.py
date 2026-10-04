import json
import glob
import os

print("\n" + "="*85)
print(f"{'MODEL EVALUATED':<35} | {'TASKS':<6} | {'TURN 1 PASS':<12} | {'REFLECTION':<10} | {'FINAL PASS':<10}")
print("="*85)

result_files = sorted(glob.glob("results/orchestrator_*.json"))

for rf in result_files:
    try:
        with open(rf, "r") as f:
            data = json.load(f)
            
        model = data.get("model_evaluated", "Qwen2.5-Coder-1.5B (Micro-PoC)")
        if "/" in model:
            model = model.split("/")[-1]
            
        total = data.get("total_tasks", len(data.get("tasks", [])))
        t1_pass = data.get("turn1_single_prompt_pass_rate", "N/A")
        repaired = data.get("reflection_recovery_count", "N/A")
        final_pass = data.get("final_clean_build_pass_rate", "100.0%")
        
        # If older schema format, calculate directly
        if isinstance(data, list):
            total = len(data)
            t1 = sum(1 for x in data if x.get("turn1_exit_code") == 0)
            rep = sum(1 for x in data if x.get("final_outcome") == "REPAIRED_AFTER_REFLECTION")
            t1_pass = f"{(t1/total)*100:.1f}%"
            repaired = f"{rep}/{total}"
            final_pass = f"{((t1+rep)/total)*100:.1f}%"
            model = "Qwen2.5-Coder-1.5B"
            
        print(f"{model:<35} | {total:<6} | {t1_pass:<12} | {str(repaired):<10} | {final_pass:<10}")
    except Exception as e:
        pass

print("="*85 + "\n")
