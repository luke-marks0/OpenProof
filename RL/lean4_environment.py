import subprocess
import tempfile
import os
import logging
import re

logging.basicConfig(level=logging.DEBUG)


class Lean4Environment:
    def __init__(self, formal_statement, state_before, lean_path='lean'):
        self.formal_statement = formal_statement
        self.state_before = state_before
        self.current_proof = state_before
        self.lean_path = lean_path
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.lean', delete=False)

    def reset(self):
        self.current_proof = self.state_before
        return self.current_proof

    def step(self, tactic):
        lean_code = f"""
    theorem my_theorem : {self.formal_statement} := by
      {self.current_proof}
      {tactic}
    """
        with open(self.temp_file.name, 'w') as f:
            f.write(lean_code)

        process = subprocess.run(
            [self.lean_path, self.temp_file.name, '--json'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        output = process.stdout + process.stderr
        logging.debug(f"Lean output:\n{output}")

        if process.returncode == 0:
            reward, done, next_state = 1.0, True, None
        else:
            next_state = self.extract_goal_state(output)
            if next_state:
                reward, done = 0.0, False
                self.current_proof += f"\n  {tactic}"
            else:
                logging.warning("Failed to extract goal state.")
                reward, done = -1.0, True

        return next_state, reward, done

    def extract_goal_state(self, output):
        goal_pattern = re.compile(r'(?<=\bgoal\s)(.*)', re.DOTALL)
        match = goal_pattern.search(output)
        return match.group(0).strip() if match else None

    def render(self):
        print("Current Proof State:")
        print(self.current_proof)

    def close(self):
        self.temp_file.close()
        os.unlink(self.temp_file.name)

