# main.py
from evolution_fast import evolve, print_results


def get_user_target():
    num_inputs = int(input("Enter number of inputs (2 or 3): "))
    num_outputs = int(input("Enter number of outputs (1 or 2): "))

    if num_inputs == 2:
        inputs = [(a, b, 0) for a in [0, 1] for b in [0, 1]]
    else:
        inputs = [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]

    print("\nInput rows order:")
    for row in inputs:
        print(row)

    targets = []
    for o in range(num_outputs):
        print(f"\nEnter output truth table #{o+1} ({len(inputs)} values):")
        values = list(map(int, input("→ ").split()))
        if len(values) != len(inputs):
            raise ValueError("❌ Incorrect number of truth table entries.")
        targets.append(values)

    return num_inputs, num_outputs, inputs, targets


if __name__ == "__main__":
    num_inputs, num_outputs, inputs, targets = get_user_target()
    best, score, max_score = evolve(num_inputs, num_outputs, inputs, targets)
    print_results(best, score, max_score, num_outputs, inputs, targets)
