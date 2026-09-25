import json
from collections import defaultdict

print("=== Sampling 25 Diverse Pure-Refactoring Tasks ===")

with open("data/SWE-Refactor/pure_refactoring_data.json", "r") as f:
    data = json.load(f)

# Filter for pure refactoring tasks that compiled successfully and have coverage info
valid_tasks = [
    t for t in data
    if t.get("isPureRefactoring") is True
    and t.get("compileResultCurrent") is True
    and t.get("coverageInfo") is not None
]

print(f"Total valid tasks meeting strict criteria: {len(valid_tasks)}")

# Group tasks by project to enforce cross-repository balance
project_buckets = defaultdict(list)
for t in valid_tasks:
    project_buckets[t.get("projectName")].append(t)

print(f"Distinct projects available: {len(project_buckets)}")

# Round-robin selection: pick max 2-3 per project up to 25 tasks
selected_tasks = []
target_count = 25

while len(selected_tasks) < target_count:
    added_in_round = False
    for proj in sorted(project_buckets.keys()):
        if project_buckets[proj]:
            t = project_buckets[proj].pop(0)
            selected_tasks.append({
                "uniqueId": t.get("uniqueId"),
                "project": t.get("projectName"),
                "refactoringType": t.get("type"),
                "compileJDK": t.get("compileJDK"),
                "compileCommand": t.get("compileCommand"),
                "filePathBefore": t.get("filePathBefore"),
                "methodNameBefore": t.get("methodNameBefore"),
                "sourceCodeBefore": t.get("sourceCodeBeforeRefactoring"),
                "groundTruthAfter": t.get("sourceCodeAfterRefactoring"),
                "coverageInfo": t.get("coverageInfo")
            })
            added_in_round = True
            if len(selected_tasks) == target_count:
                break
    if not added_in_round:
        break

with open("data/poc_25_tasks.json", "w") as f:
    json.dump(selected_tasks, f, indent=2)

print(f"\n--- Selected {len(selected_tasks)} Balanced Tasks across {len(set(t['project'] for t in selected_tasks))} Projects ---")
project_counts = defaultdict(int)
for st in selected_tasks:
    project_counts[st['project']] += 1

for proj, count in sorted(project_counts.items()):
    print(f"  • {proj}: {count} tasks")

print("\nSaved to data/poc_25_tasks.json")
