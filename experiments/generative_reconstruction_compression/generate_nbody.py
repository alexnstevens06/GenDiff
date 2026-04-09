import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from experiments.generative_reconstruction_compression.openrouter_client import OpenRouterClient

PROMPT = """
Write a comprehensive, professional, 500-line N-body simulation in Python using the Pygame library.

Requirements:
1. The simulation must allow the user to add new astronomical objects dynamically by clicking the mouse. The mass / velocity of the new object should be determined by mouse drag or click duration.
2. The visuals must be variable and beautiful: varying colors based on mass or velocity, dynamic trails showing previous paths, and a star-field background.
3. The objects must have varying masses, affecting their gravitational pull and visual size.
4. Implement a robust numerical integration method (e.g., Runge-Kutta 4th Order or Velocity Verlet) to ensure stable orbits.
5. Include a UI overlay (text on screen) showing the total number of bodies, current FPS, and controls.
6. Handle collisions: if two bodies get too close, they should either merge into a larger body (conserving momentum) or explicitly bounce.
7. Include comprehensive docstrings and comments. Make the code modular with clean classes (e.g., `Body`, `Simulation`, `UI`).
8. The total file length must be around 100 lines. Please flesh out the physics, rendering, and interaction logic extensively to meet this length requirement. Provide ONLY valid Python code without any markdown wrappers.

Start with standard imports and go from there.
"""

SEED = 8675309
MODEL = "moonshotai/kimi-k2.5" 

def main():
    print(f"Instantiating OpenRouterClient with model...")
    # Trying the user's specific text or defaulting to moonshot 32k since kimi-k2.5 might not be standard on OR
    client = OpenRouterClient(model=MODEL)
    
    print("Generating simulation via prompt...")
    code = client.generate(PROMPT, seed=SEED, temperature=0.7)
    
    if not code:
        print("Failed to generate code.")
        return
        
    # Strip markdown block if it answered with it anyway
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
        
    code = code.strip()

    # Save prompt
    with open("experiments/generative_reconstructive_compression/nbody_prompt.txt", "w") as f:
        f.write(PROMPT)
        
    # Save seed    
    with open("experiments/generative_reconstructive_compression/nbody_seed.txt", "w") as f:
        f.write(str(SEED))
        
    # Save code    
    with open("experiments/generative_reconstructive_compression/nbody_candidate.py", "w") as f:
        f.write(code)
        
    print(f"Generated {len(code.splitlines())} lines of code.")
    print("Saved nbody_candidate.py, nbody_prompt.txt, and nbody_seed.txt.")

if __name__ == "__main__":
    main()
