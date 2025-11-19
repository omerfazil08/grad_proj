# main_phase5b.py
# Driver script for Phase 5b Adaptive Macro Decay Evolution

from evolution_phase5b import GAConfig, evolve_phase5b
from grad_34_simp6 import simplify_single_output

# ----------------- helper -----------------
def generate_inputs(n):
    combos = []
    for i in range(2 ** n):
        bits = [(i >> (n - 1 - b)) & 1 for b in range(n)]
        combos.append(tuple(bits))
    return combos

# ----------------- user interface -----------------
def get_user_target():
    num_inputs = int(input("Enter number of inputs (2..8): "))
    num_outputs = int(input("Enter number of outputs (1..4): "))
    inputs = generate_inputs(num_inputs)
    print("\nInput rows order:")
    for row in inputs: print(row)

    targets = []
    for i in range(num_outputs):
        vals = list(map(int, input(f"\nEnter output truth table #{i+1} ({len(inputs)} values):\n→ ").split()))
        if len(vals) != len(inputs):
            raise ValueError("Incorrect number of truth values.")
        targets.append(vals)
    targets = [list(x) for x in zip(*targets)]
    return num_inputs, num_outputs, inputs, targets


# ----------------- main -----------------
if __name__ == "__main__":
    num_inputs, num_outputs, inputs, targets = get_user_target()

    cfg = GAConfig(
        pop_size=800,
        generations=1200,
        macro_decay_start=0,
        macro_decay_end=120,
        p_choose_primitive=0.3
    )

    best, score, max_score = evolve_phase5b(num_inputs, num_outputs, inputs, targets, cfg)
    print(f"\n✅ Best Network Found: {score} / {max_score}")

    for g in best:
        print(f"{g['name']}: {g['gate']}({', '.join(g['inputs'])})")

 # show simplified logic for all outputs
print("\n🧠 Simplified Output Logic:")
for j in range(num_outputs):
    out_gate = best[-1 - j]["name"]
    expr = simplify_single_output(best, out_gate)
    print(f"Output {j+1} ({out_gate}): {expr}")
