"""Quick test: Verify frog predator is hunting and moving correctly."""

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Frog_predator_neuro.simulation import Simulation


def test_frog_predator():
    """Launch frog simulation and check hunting behavior."""
    print("=== Frog Predator Neuro - Quick Test ===")
    print("Testing frog hunting behavior for 20 seconds...\n")

    sim = Simulation(headless=True)
    try:
        agent = sim.agents[0] if sim.agents else None

        if not agent:
            print("ERROR: No agents spawned!")
            return False

        for frame in range(2000):
            dt = 1.0 / 100.0
            sim.update(dt)
            if not sim.headless:
                sim.draw()

            if frame % 500 == 0 and frame > 0:
                time_s = frame / 100.0
                energy = agent.body_energy
                food_collected = agent.food_collected
                distance = agent.last_step_distance if hasattr(agent, "last_step_distance") else 0.0
                x, y = agent.x, agent.y

                print(
                    f"[{time_s:6.1f}s] Energy={energy:.3f} | Food={food_collected} | "
                    f"Pos=({x:7.1f}, {y:7.1f}) | Move={distance:.2f}"
                )

        print("\n=== Results ===")
        print(f"Final energy: {agent.body_energy:.3f}")
        print(f"Food collected: {agent.food_collected}")

        if agent.food_collected > 0:
            print("\nSUCCESS - Frog is hunting and eating!")
            return True
        if agent.food_collected == 0 and agent.body_energy < 0.6:
            print("\nPARTIAL - Frog is moving but not capturing prey")
            return False

        print("\nFAIL - Frog not moving or hunting")
        return False
    finally:
        sim.close()


if __name__ == "__main__":
    success = test_frog_predator()
    raise SystemExit(0 if success else 1)
