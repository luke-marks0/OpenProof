import subprocess
import tempfile
import os
import logging
import re
import textwrap
from typing import Tuple, Optional, Dict
import shutil
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)


@dataclass
class RewardConfig:
    """Configuration for reward shaping"""
    proof_complete: float = 1.0
    proof_failed: float = -1.0
    progress_base: float = 0.0
    progress_complexity_reduction: float = 0.1
    progress_new_hypothesis: float = 0.05
    sorry_penalty: float = -0.5
    trivial_step_penalty: float = -0.1


class Lean4Environment:
    """A Lean 4 theorem proving environment."""
    
    def __init__(self, formal_statement: str, lean_path: Optional[str] = None, reward_config: Optional[RewardConfig] = None):
        """
        Initialize the Lean environment.
        
        Args:
            formal_statement: The theorem statement to prove
            lean_path: Optional path to Lean executable
            reward_config: Optional configuration for reward shaping
        """
        self.formal_statement = self._sanitize_statement(formal_statement)
        self.current_proof = ""
        self.lean_path = self._find_lean_executable(lean_path)
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.lean', delete=False, encoding='utf-8')
        self.last_goal_state = None
        self.reward_config = reward_config or RewardConfig()
        
    def _sanitize_statement(self, statement: str) -> str:
        return statement.strip()
        
    def _find_lean_executable(self, provided_path: Optional[str] = None) -> str:
        if provided_path and os.path.isfile(provided_path):
            return provided_path
            
        lean_path = shutil.which('lean')
        if lean_path:
            return lean_path
            
        raise FileNotFoundError("Could not find Lean executable")
        
    def _calculate_state_complexity(self, state: str) -> int:
        """Calculate complexity of a goal state."""
        operators = ['+', '-', '*', '/', '^', '∧', '∨', '→', '↔', '=', '≤', '≥', '<', '>']
        return len(state) + sum(state.count(op) for op in operators)
        
    def _calculate_progress_reward(self, current_state: str, tactic: str) -> float:
        """Calculate reward based on proof progress."""
        if 'sorry' in tactic:
            return self.reward_config.sorry_penalty
            
        if tactic.strip() in ['', 'skip', 'trivial']:
            return self.reward_config.trivial_step_penalty
            
        if not self.last_goal_state:
            return self.reward_config.progress_base
            
        reward = self.reward_config.progress_base
        
        current_complexity = self._calculate_state_complexity(current_state)
        previous_complexity = self._calculate_state_complexity(self.last_goal_state)
        
        if current_complexity < previous_complexity:
            reward += self.reward_config.progress_complexity_reduction
            
        current_hypotheses = len(current_state.split('\n'))
        previous_hypotheses = len(self.last_goal_state.split('\n'))
        
        if current_hypotheses > previous_hypotheses:
            reward += self.reward_config.progress_new_hypothesis
            
        return reward

    def step(self, tactic: str) -> Tuple[Optional[str], float, bool]:
        """
        Apply a tactic and return the new state, reward, and whether proof is complete.
        
        Args:
            tactic: The tactic to apply
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        lean_code = textwrap.dedent(f"""\
            import Lean.Elab.Tactic
            
            theorem my_theorem {self.formal_statement} := by
              {self.current_proof}
              {tactic}
            """)
            
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            f.write(lean_code)
            
        process = subprocess.run(
            [self.lean_path, self.temp_file.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )
            
        output = process.stdout + process.stderr
            
        if process.returncode == 0:
            if tactic.strip():
                logging.info(f"✓ Proof completed with: {tactic}")
            return None, self.reward_config.proof_complete, True
                
        next_state = self.extract_goal_state(output)
        if next_state:
            reward = self._calculate_progress_reward(next_state, tactic)
            if reward > self.reward_config.progress_base and tactic.strip():
                logging.info(f"→ Progress with {tactic}: {next_state}")
            self.current_proof += f"\n  {tactic}"
            self.last_goal_state = next_state
            return next_state, reward, False
                
        return None, self.reward_config.proof_failed, True

    def extract_goal_state(self, output: str) -> Optional[str]:
        """Extract the current goal state from Lean output."""
        patterns = [
            r'⊢\s+(.*?)(?:\n|$)',
            r'goal:\s*(.*?)(?:\n|$)',
            r'type:\s*(.*?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.MULTILINE)
            if match:
                return match.group(1).strip()
                
        return None

    def reset(self) -> str:
        """Reset the environment to initial state."""
        self.current_proof = ""
        self.last_goal_state = None
        return self.current_proof

    def render(self):
        """Print the current proof state."""
        print("Current Proof State:")
        print(self.current_proof)

    def close(self):
        """Clean up resources."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
