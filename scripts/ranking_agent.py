"""
GWU D.Eng. Research Praxis: Ranking Agent (Pairwise Tournament Engine)
Author: Ahmad Obiedat | Advisor: Professor Joseph McEttrick
Implements Pairwise Elo Ranking to eliminate positional bias (Gottweis et al., 2025).
"""

import math
import random
from typing import Dict, Any, Tuple

class TournamentRankingAgent:
    def __init__(self, k_factor: float = 32.0, base_elo: float = 1200.0):
        self.k_factor = k_factor
        self.base_elo = base_elo
        self.ratings = {}

    def get_rating(self, candidate_id: str) -> float:
        return self.ratings.get(candidate_id, self.base_elo)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Standard logistic Elo expected score formula."""
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    def evaluate_pairwise_match(self, patch_a: Dict[str, Any], patch_b: Dict[str, Any]) -> float:
        """
        Deterministic scoring function evaluating candidate refactorings:
        1. Clean compilation (Exit code 0) = +50 points
        2. Lower Cyclomatic Complexity V(G) = +25 points
        3. Zero test tampering = +25 points
        Returns: 1.0 if A wins, 0.0 if B wins, 0.5 if tie.
        """
        score_a = 0
        score_b = 0

        # Criterion 1: Compilation validity
        if patch_a.get("turn1_exit_code") == 0 or patch_a.get("final_outcome") == "REPAIRED_AFTER_REFLECTION":
            score_a += 50
        if patch_b.get("turn1_exit_code") == 0 or patch_b.get("final_outcome") == "REPAIRED_AFTER_REFLECTION":
            score_b += 50

        # Criterion 2: Cyclomatic complexity reduction
        comp_a = patch_a.get("complexity_after", 10)
        comp_b = patch_b.get("complexity_after", 10)
        if comp_a < comp_b:
            score_a += 25
        elif comp_b < comp_a:
            score_b += 25
        else:
            score_a += 12.5
            score_b += 12.5

        # Criterion 3: Immutability verified
        if patch_a.get("test_suite_immutability_verified", True):
            score_a += 25
        if patch_b.get("test_suite_immutability_verified", True):
            score_b += 25

        if score_a > score_b:
            return 1.0
        elif score_b > score_a:
            return 0.0
        return 0.5

    def update_elo(self, id_a: str, id_b: str, result_a: float):
        """Updates Elo ratings based on match outcome."""
        r_a = self.get_rating(id_a)
        r_b = self.get_rating(id_b)

        exp_a = self.expected_score(r_a, r_b)
        exp_b = 1.0 - exp_a

        self.ratings[id_a] = round(r_a + self.k_factor * (result_a - exp_a), 2)
        self.ratings[id_b] = round(r_b + self.k_factor * ((1.0 - result_a) - exp_b), 2)

    def run_tournament(self, candidates: list) -> list:
        """Executes a round-robin tournament across all candidate patches."""
        ids = [c["uniqueId"] for c in candidates]
        cand_map = {c["uniqueId"]: c for c in candidates}

        # Randomized pairwise matches
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                # Swap presentation order randomly to eliminate positional bias
                if random.random() > 0.5:
                    outcome = self.evaluate_pairwise_match(cand_map[id_a], cand_map[id_b])
                    self.update_elo(id_a, id_b, outcome)
                else:
                    outcome = self.evaluate_pairwise_match(cand_map[id_b], cand_map[id_a])
                    self.update_elo(id_b, id_a, outcome)

        # Return candidates sorted by Elo rating
        ranked = sorted(candidates, key=lambda c: self.get_rating(c["uniqueId"]), reverse=True)
        return ranked

if __name__ == "__main__":
    agent = TournamentRankingAgent()
    sample_a = {"uniqueId": "Patch_A_SinglePrompt", "complexity_after": 5, "final_outcome": "FAIL"}
    sample_b = {"uniqueId": "Patch_B_Reflection", "complexity_after": 3, "final_outcome": "REPAIRED_AFTER_REFLECTION"}

    print("Running Pairwise Tournament Ranking Test...")
    ranked = agent.run_tournament([sample_a, sample_b])
    for rank, p in enumerate(ranked, 1):
        print(f"Rank {rank}: {p['uniqueId']} | Elo: {agent.get_rating(p['uniqueId'])}")
