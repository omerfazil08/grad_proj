# main_phase5a_turbo.py
from evolution_phase5a_turbo import (
    TurboConfig, evolve_5a_turbo, print_results
)

# Optional truth-table simplifier that is bit-accurate (if you created grad_34_simp7.py)
try:
    from grad_34_simp7 import simplify_single_output
    HAVE_SIMP = True
except Exception:
    HAVE_SIMP = False

def get_user_target():
    num_inputs  = int(input("Enter number of inputs (2..8): ").strip())
    num_outputs = int(input("Enter number of outputs (1..4): ").strip())

    inputs = []
    for i in range(1 << num_inputs):
        row = tuple((i >> (num_inputs - 1 - b)) & 1 for b in range(num_inputs))
        inputs.append(row)

    print("\nInput rows order:")
    for r in inputs:
        print(r)

    targets = []
    for o in range(num_outputs):
        print(f"\nEnter output truth table #{o+1} ({len(inputs)} values):")
        vals = list(map(int, input("→ ").split()))
        if len(vals) != len(inputs):
            raise ValueError("Incorrect number of values.")
        targets.append(vals)

    return num_inputs, num_outputs, inputs, targets

if __name__ == "__main__":
    n_in, n_out, inputs, targets = get_user_target()

    cfg = TurboConfig(
        num_gates=16,
        pop_size=800,
        generations=1200,
        elitism=10,
        tournament_k=5,
        base_mut=0.30,
        min_mut=0.06,
        p_choose_primitive=0.70,
        diversity_every=800,
        diversity_fraction=0.10,
        local_search_on_elite=True,
        local_search_elites=6,
        local_search_trials=2,
        log_every=20,
        seed=42,
        size_penalty_lambda=0.0,
    )

    best, score, max_score = evolve_5a_turbo(n_in, n_out, inputs, targets, cfg)

    print_results(best, score, max_score, n_out, inputs, targets)

    if HAVE_SIMP:
        print("\n🧠 Truth-table–exact simplified outputs:")
        base_idx = len(best) - n_out
        out_names = [best[base_idx + o]["name"] for o in range(n_out)]
        var_order = [chr(ord('A') + i) for i in range(n_in)]
        for i, on in enumerate(out_names, 1):
            expr = simplify_single_output(best, on, input_names=var_order)
            print(f"Output {i} ({on}):  {expr}")
    else:
        print("\n(Note) grad_34_simp7.py not found; skipping symbolic simplification.")
