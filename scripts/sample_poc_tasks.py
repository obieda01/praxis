import json

with open("data/SWE-Refactor/pure_refactoring_data.json", "r") as f:
    data = json.load(f)

# Filter for pure refactoring tasks that compiled successfully and have coverage info
valid_tasks = [
    t for t in data
    if t.get("isPureRefactoring") is True
    and t.get("compileResultCurrent") is True
    and t.get("coverageInfo") is not None
]

print(f"Total valid tasks meeting compilation & coverage gates: {len(valid_tasks)}")

# Select 5 tasks across distinct projects and refactoring types
selected_tasks = []
seen_types = set()
seen_projects = set()

for t in valid_tasks:
    ref_type = t.get("type")
    proj = t.get("projectName")
    if (ref_type not in seen_types or proj not in seen_projects) or len(selected_tasks) < 5:
        selected_tasks.append({
            "uniqueId": t.get("uniqueId"),
            "project": proj,
            "refactoringType": ref_type,
            "compileJDK": t.get("compileJDK"),
            "compileCommand": t.get("compileCommand"),
            "filePathBefore": t.get("filePathBefore"),
            "methodNameBefore": t.get("methodNameBefore"),
            "sourceCodeBefore": t.get("sourceCodeBeforeRefactoring"),
            "groundTruthAfter": t.get("sourceCodeAfterRefactoring"),
            "coverageInfo": t.get("coverageInfo")
        })
        seen_types.add(ref_type)
        seen_projects.add(proj)
        if len(selected_tasks) == 5:
            break

with open("data/poc_5_tasks.json", "w") as f:
    json.dump(selected_tasks, f, indent=2)

print("\n--- Selected 5 Micro-PoC Tasks ---")
for idx, st in enumerate(selected_tasks, 1):
    print(f"{idx}. [{st['project']}] {st['refactoringType']} (JDK {st['compileJDK']})")
    print(f"   Method: {st['methodNameBefore']}")
    print(f"   Compile: {st['compileCommand'][:65]}...")
print("\nSaved to data/poc_5_tasks.json")
