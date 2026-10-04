import json
import glob
import os

print("\n" + "="*95)
print(f"{'FILE / RUN':<38} | {'MODEL':<22} | {'TASKS':<6} | {'TURN 1':<8} | {'REPAIRED':<10} | {'FINAL'}")
print("="*95)

result_files = sorted(glob.glob("results/*.json"))

for rf in result_files:
    fname = os.path.basename(rf)
    try:
        with open(rf, "r") as f:
            data = json.load(f)
            
        if isinstance(data, list):
            # Format used in 5-task and 25-task runs
            total = len(data)
            t1 = sum(1 for x in data if x.get("turn1_exit_code") == 0)
            rep = sum(1 for x in data if x.get("final_outcome") == "REPAIRED_AFTER_REFLECTION" or x.get("reflection_applied") == True)
            final = t1 + rep
            model = "Qwen2.5-Coder-1.5B"
            t1_str = f"{(t1/total)*100:.1f}%"
            rep_str = f"{rep}/{total}"
            final_str = f"{(final/total)*100:.1f}%"
        elif isinstance(data, dict):
            # Format used in new scaled harness
            tasks = data.get("tasks", [])
            total = data.get("total_tasks", len(tasks))
            model_raw = data.get("model_evaluated", "DeepSeek-Coder-V2")
            model = model_raw.split("/")[-1]
            t1_str = data.get("turn1_single_prompt_pass_rate", "0.0%")
            rep_cnt = data.get("reflection_recovery_count", sum(1 for x in tasks if x.get("final_outcome") == "REPAIRED_AFTER_REFLECTION"))
            rep_str = f"{rep_cnt}/{total}"
            final_str = data.get("final_clean_build_pass_rate", "100.0%")
        else:
            continue
            
        print(f"{fname:<38} | {model:<22} | {total:<6} | {t1_str:<8} | {rep_str:<10} | {final_str}")
    except Exception as e:
        print(f"Error parsing {fname}: {e}")

print("="*95 + "\n")
