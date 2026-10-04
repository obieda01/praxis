import json
import os
import random

dataset_path = "data/SWE-Refactor/pure_refactoring_data.json"
output_path = "data/poc_50_tasks.json"

if os.path.exists(dataset_path):
    print(f"Loading tasks from {dataset_path}...")
    with open(dataset_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} total tasks.")
    
    # Stratified sample across all available projects
    by_project = {}
    for item in data:
        p = item.get("project", "unknown")
        by_project.setdefault(p, []).append(item)
    
    sampled = []
    projects = list(by_project.keys())
    random.seed(42)
    
    # Pick evenly across projects
    while len(sampled) < 50:
        for p in projects:
            if by_project[p]:
                sampled.append(by_project[p].pop(0))
            if len(sampled) == 50:
                break
else:
    print("Warning: pure_refactoring_data.json not found, expanding from poc_25_tasks.json...")
    with open("data/poc_25_tasks.json", "r") as f:
        sampled = json.load(f)
    # Expand to 50
    sampled = (sampled * 2)[:50]

with open(output_path, "w") as f:
    json.dump(sampled, f, indent=2)

print(f"Successfully generated {len(sampled)} tasks in {output_path}")
