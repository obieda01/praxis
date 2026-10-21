"""
GWU D.Eng. Research Praxis: Evolution Agent
Author: Ahmad Obiedat | Advisor: Professor Joseph McEttrick
Optimizes McCabe cyclomatic complexity across iterative evolution loops (Hypothesis 2).
"""

import re

def compute_cyclomatic_complexity(code: str) -> int:
    """
    Computes McCabe Cyclomatic Complexity (M = Decision Points + 1)
    for Java source snippets by stripping comments/strings and counting branching primitives.
    """
    if not code:
        return 1
    clean_code = re.sub(r'//.*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"', '', code)
    branch_patterns = [
        r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\bcase\b', r'\bcatch\b',
        r'\&\&', r'\|\|', r'\?'
    ]
    complexity = 1
    for pat in branch_patterns:
        complexity += len(re.findall(pat, clean_code))
    return complexity

class EvolutionAgent:
    def __init__(self):
        self.system_prompt = (
            "You are a Java Code Evolution Agent specialized in reducing McCabe cyclomatic complexity. "
            "Refactor the provided Java code to simplify nested conditionals, replace verbose if/else blocks "
            "with guard clauses or polymorphic/stream expressions, and eliminate redundant boolean branches "
            "while strictly preserving exact method signatures, behavior, and compilation validity."
        )

    def apply_ast_simplifications(self, java_code: str, loop_idx: int) -> str:
        """Applies deterministic non-destructive AST branch simplifications across evolution loops."""
        code = java_code
        # Loop 1: Simplify boolean return ladders (if (cond) return true; else return false; -> return cond;)
        code = re.sub(
            r'if\s*\(([^)]+)\)\s*\{\s*return\s+true;\s*\}\s*else\s*\{\s*return\s+false;\s*\}',
            r'return \1;',
            code
        )
        # Loop 2: Collapse null-check throw ladders into Objects.requireNonNull guard clauses
        if loop_idx >= 2:
            code = re.sub(
                r'if\s*\(([^=!]+)\s*==\s*null\)\s*\{\s*throw\s+new\s+\w+Exception\([^)]*\);\s*\}',
                r'java.util.Objects.requireNonNull(\1);',
                code
            )
            # Remove redundant else blocks after return statements
            code = re.sub(r'(\breturn\b[^;]+;\s*\})\s*else\s*\{([^}]*)\}', r'\1\n\2', code)
        # Loop 3: Consolidate nested single-statement if conditions and strip defensive scaffolding
        if loop_idx >= 3:
            code = re.sub(r'\s*&&\s*true\b', '', code)
            code = re.sub(r'\s*\|\|\s*false\b', '', code)
            # Flatten duplicate defensive null checks added during Turn 2 reflection repair
            lines = code.splitlines()
            cleaned = []
            seen_guards = set()
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("if (") and "== null" in stripped:
                    if stripped in seen_guards:
                        continue
                    seen_guards.add(stripped)
                cleaned.append(line)
            code = "\n".join(cleaned)
        return code

if __name__ == "__main__":
    sample = """
    public boolean validate(String input) {
        if (input == null) { throw new IllegalArgumentException("null"); }
        if (input.isEmpty() && true) { return true; } else { return false; }
    }
    """
    agent = EvolutionAgent()
    c0 = compute_cyclomatic_complexity(sample)
    e3 = agent.apply_ast_simplifications(sample, loop_idx=3)
    c3 = compute_cyclomatic_complexity(e3)
    print(f"Evolution Agent Verified | Complexity Before: {c0} -> Complexity After (Loop 3): {c3}")
