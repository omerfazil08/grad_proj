# main_colab_phase0.py
#
# Simple driver for evolution_colab_phase0.py
# - Has an interactive mode (like your old main*) using input()
# - Also has a ready-to-run XOR demo for Colab: run_demo_xor()

from evolution_colab_phase1 import (
    ColabConfig,
    evolve_colab_phase0,
    print_results,
)


def get_user_target():
    """Interactive prompt (works both locally and in Colab cells)."""
    num_inputs = int(input("Enter number of inputs (2..8): ").strip())
    num_outputs = int(input("Enter number of outputs (1..4): ").strip())

    print("\nInput rows order will be standard binary counting:")
    rows = []
    for i in range(2 ** num_inputs):
        tup = tuple((i >> b) & 1 for b in range(num_inputs - 1, -1, -1))
        rows.append(tup)
    for idx, r in enumerate(rows):
        print(f"{idx:2d}: {r}")
    print()

    targets = []
    for o in range(num_outputs):
        vals = input(f"Enter output truth table #{o+1} "
                     f"({len(rows)} values 0/1 separated by spaces):\n→ ").strip().split()
        if len(vals) != len(rows):
            raise ValueError(f"Expected {len(rows)} values, got {len(vals)}")
        col = [int(v) for v in vals]
        targets.append(col)

    return num_inputs, num_outputs, rows, targets


def run_ga_interactive():
    """Interactive GA run using the user-provided truth table."""
    n_in, n_out, inputs, targets = get_user_target()

    cfg = ColabConfig(
        num_gates=16,
        pop_size=400,
        generations=800,
        elitism=6,
        tournament_k=4,
        base_mut=0.30,
        min_mut=0.06,
        p_choose_primitive=0.70,
        log_every=20,
        record_history=True,
        seed=42,
        size_penalty_lambda=0.0,
    )

    best, score, max_score, history = evolve_colab_phase0(
        n_in, n_out, inputs, targets, cfg
    )

    print_results(best, score, max_score, n_out, inputs, targets)

    # (Optional) history is returned for plotting in Colab:
    # e.g.:
    #   import matplotlib.pyplot as plt
    #   plt.plot(history["gen"], history["best"])
    #   plt.plot(history["gen"], history["avg"])
    #   plt.show()

def run_demo_full_adder():
    """
    Demo: 4-input, 3-output Full Adder-like truth table.
    Uses the 16-row truth table you provided.
    """

    # 4 inputs → 16 rows
    n_in = 4
    n_out = 3

    # All input combinations: binary count from 0000 to 1111
    inputs = [( (i >> 3) & 1,
                (i >> 2) & 1,
                (i >> 1) & 1,
                (i >> 0) & 1 ) for i in range(16)]

    # Outputs you provided:
    S1 =  [0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1]
    S0 =  [0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0]
    Cout = [0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1]

    targets = [S1, S0, Cout]

    cfg = ColabConfig(
        num_gates=18,
        pop_size=400,
        generations=800,
        elitism=6,
        tournament_k=4,
        base_mut=0.35,
        min_mut=0.10,
        p_choose_primitive=0.70,
        log_every=20,
        record_history=True,
        seed=42,
        size_penalty_lambda=0.0,

    )

    best, score, max_score, history = evolve_colab_phase0(
        n_in, n_out, inputs, targets, cfg
    )

    print_results(best, score, max_score, n_out, inputs, targets)

    return best, score, max_score, history


def run_demo_xor():
    """
    Non-interactive demo: 2-input XOR with 1 output.
    Ideal to just run in Colab without typing anything.
    """
    n_in = 2
    n_out = 1
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    targets = [[0, 1, 1, 0]]  # XOR

    cfg = ColabConfig(
        num_gates=12,
        pop_size=300,
        generations=500,
        elitism=4,
        tournament_k=3,
        base_mut=0.35,
        min_mut=0.10,
        p_choose_primitive=0.70,
        log_every=10,
        record_history=True,
        seed=123,
        size_penalty_lambda=0.0,
    )

    best, score, max_score, history = evolve_colab_phase0(
        n_in, n_out, inputs, targets, cfg
    )

    print_results(best, score, max_score, n_out, inputs, targets)

    # Example plotting usage (in Colab):
    # import matplotlib.pyplot as plt
    # plt.plot(history["gen"], history["best"])
    # plt.plot(history["gen"], history["avg"])
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness (bits correct)")
    # plt.legend(["Best", "Average"])
    # plt.grid(True)
    # plt.show()

    return best, score, max_score, history



if __name__ == "__main__":
    # For local runs you can choose:
    # 1) interactive mode, or
    # 2) XOR demo
    print("1) Interactive GA run")
    print("2) XOR demo")
    print("3) 4-input 3-output Full-Adder demo")
    choice = input("Select option (1/2/3): ").strip()

    if choice == "1":
        run_ga_interactive()
    elif choice == "3":
        run_demo_full_adder()
    else:
        run_demo_xor()