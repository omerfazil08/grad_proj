# main_colab_phase5.py
import time
import sys

try:
    import evolution_colab_phase5 as engine
except ImportError:
    print("❌ Error: 'evolution_colab_phase5.py' not found.")
    sys.exit(1)

def get_user_target():
    try:
        num_inputs = int(input("Enter number of inputs (2..12): ").strip())
        num_outputs = int(input("Enter number of outputs (1..10): ").strip())
    except ValueError:
        num_inputs, num_outputs = 2, 1
    
    rows = []
    for i in range(2 ** num_inputs):
        tup = tuple((i >> b) & 1 for b in range(num_inputs - 1, -1, -1))
        rows.append(tup)
        
    targets = []
    print(f"Input rows: {len(rows)}")
    for o in range(num_outputs):
        val_str = input(f"Enter output truth table #{o+1} ({len(rows)} values):\n→ ").strip()
        vals = [int(v) for v in val_str.split()]
        if len(vals) < len(rows):
             vals += [0] * (len(rows) - len(vals))
        targets.append(vals[:len(rows)])
    return num_inputs, num_outputs, rows, targets

def evaluate_serial(individual, input_tuple):
    """Serial verification."""
    signals = list(input_tuple)
    for gate in individual:
        gtype = gate['type']
        ins = gate['inputs']
        vals = [signals[i] if i < len(signals) else 0 for i in ins]
        
        res = 0
        # Basic
        if gtype == "AND": res = vals[0] & vals[1]
        elif gtype == "OR":  res = vals[0] | vals[1]
        elif gtype == "XOR" or gtype == "XOR2": res = vals[0] ^ vals[1]
        elif gtype == "NOT": res = 1 - vals[0]
        elif gtype == "NAND": res = 1 - (vals[0] & vals[1])
        elif gtype == "NOR":  res = 1 - (vals[0] | vals[1])
        elif gtype == "XNOR" or gtype == "XNOR2": res = 1 - (vals[0] ^ vals[1])
        # Macros
        elif gtype == "MUX2":
            # b_MUX2(s, a, b) -> if s=0 then a, else b
            # args: vals[0]=s, vals[1]=a, vals[2]=b
            s, a, b = vals[0], vals[1], vals[2]
            res = a if s == 0 else b
        elif gtype == "HALF_SUM": res = vals[0] ^ vals[1]
        elif gtype == "HALF_CARRY": res = vals[0] & vals[1]
        elif gtype == "FULL_SUM": res = vals[0] ^ vals[1] ^ vals[2]
        elif gtype == "FULL_CARRY":
            a, b, c = vals[0], vals[1], vals[2]
            res = (a & b) | (a & c) | (b & c)
        elif gtype == "EQ1": res = 1 - (vals[0] ^ vals[1])
        elif gtype == "GT1": res = 1 if vals[0] > vals[1] else 0
        
        signals.append(res & 1)
    return signals

def print_results(best_ind, num_inputs, num_outputs, inputs, targets):
    print("\n=== FINAL CIRCUIT ANALYSIS ===")
    if not best_ind: return

    # Gate List
    print(f"\n[Genome Structure] ({len(best_ind)} gates)")
    for i, gate in enumerate(best_ind):
        inputs_str = []
        for inp in gate['inputs']:
            if inp < num_inputs: inputs_str.append(f"I{inp}")
            else: inputs_str.append(f"g{inp - num_inputs}")
        print(f"g{i:<3} | {gate['type']:<12} | {', '.join(inputs_str)}")

    # Verification
    print("\n[Truth Table Verification]")
    header = " ".join([f"I{i}" for i in range(num_inputs)]) + " | " + \
             " ".join([f"O{i}" for i in range(num_outputs)]) + " | Target"
    print(header)
    
    correct = 0
    total = len(inputs) * num_outputs
    
    for r, row in enumerate(inputs):
        signals = evaluate_serial(best_ind, row)
        circuit_outs = signals[-num_outputs:]
        target_outs = [targets[o][r] for o in range(num_outputs)]
        
        match = (circuit_outs == target_outs)
        if match: correct += len(circuit_outs)
        else: 
            for k in range(num_outputs):
                if circuit_outs[k] == target_outs[k]: correct += 1
        
        if r < 16:
            mark = "✅" if match else "❌"
            print(f"{row} | {circuit_outs} | {target_outs} {mark}")
            
    print(f"\nFinal Accuracy: {correct}/{total} bits ({correct/total*100:.1f}%)")

def run_5bit_adder():
    n_in = 10; n_out = 6
    print(f"\n⚡ Generating 5-bit Adder Truth Table ({2**n_in} rows)...")
    inputs = []; targets = [[] for _ in range(n_out)]
    for i in range(1024):
        val_a = (i >> 5) & 0x1F
        val_b = i & 0x1F
        inputs.append(tuple((i >> b) & 1 for b in range(9, -1, -1)))
        total = val_a + val_b
        for bit in range(n_out):
            targets[bit].append((total >> bit) & 1)

    cfg = engine.Phase5Config(
        num_gates=40, pop_size=3000, generations=5000, 
        elitism=50, p_choose_primitive=0.5, log_every=50
    )
    best_ind = engine.evolve_incremental_phase5(n_in, n_out, inputs, targets, cfg)
    print_results(best_ind, n_in, n_out, inputs, targets)

if __name__ == "__main__":
    run_5bit_adder()