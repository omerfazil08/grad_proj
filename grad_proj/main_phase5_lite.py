from evolution_phase5_lite import evolve_phase5_lite

def safe_int_input(prompt, min_val, max_val):
    while True:
        val = input(prompt).strip()
        if not val.isdigit():
            print("⚠️ Enter a valid number.")
            continue
        val = int(val)
        if not (min_val <= val <= max_val):
            print(f"⚠️ Must be between {min_val} and {max_val}.")
            continue
        return val

def get_user_target():
    num_inputs = safe_int_input("Enter number of inputs (2..8): ", 2, 8)
    num_outputs = safe_int_input("Enter number of outputs (1..4): ", 1, 4)
    rows = [(tuple(int(x) for x in f"{i:0{num_inputs}b}")) for i in range(2**num_inputs)]
    print("\nInput rows order:")
    for r in rows: print(r)
    targets = []
    for j in range(num_outputs):
        out_line = input(f"\nEnter output truth table #{j+1} ({2**num_inputs} values):\n→ ").split()
        targets.append([int(x) for x in out_line])
    targets = list(zip(*targets))
    return num_inputs, num_outputs, rows, targets

if __name__ == "__main__":
    num_inputs, num_outputs, inputs, targets = get_user_target()
    best, fitness_hist, max_score = evolve_phase5_lite(inputs, targets, num_inputs, num_outputs)
    print("\n✅ Evolution complete!")
    print(f"Best achieved: {fitness_hist[-1]}/{max_score}")
    print("Saved: fitness_vs_gen.png and circuit_graph.png")
