import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Frog_predator_neuro.simulation import Simulation


def main():
    """Run biorealistic frog predator neuro game"""
    simulation = Simulation(
        width=900,
        height=700,
        num_flies=12,
        headless=False,
        training_mode=True,
        agent_count=1,  # Single frog
    )
    simulation.run(max_steps=6000)
    simulation.close()


if __name__ == "__main__":
    main()
