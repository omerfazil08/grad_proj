# main_colab_phase5.py
# Driver for Phase 5: Hybrid Engine (Bitwise + Macros + Incremental)
# No VHDL, No Simplifier - Pure Evolution & Verification.

import time
import sys

# Import the Phase 5 Engine
try:
    import evolution_colab_phase5 as engine
except ImportError:
    print("❌ Error: 'evolution_colab_phase5.py' not found. Please upload it.")
    sys.exit(1)

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def get_user_target():
    """Interactive input for custom truth tables."""
    try:
        num_inputs = int(input("Enter number of inputs (2..12): ").strip())
        num_outputs = int(input("Enter number of outputs (1..10): ").strip())
    except ValueError:
        print("Invalid input. Defaulting to 2 inputs, 1 output.")
        num_inputs, num_outputs = 2, 1
    
    # Generate all binary combinations
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
    """
    Serial evaluation for final verification (unpacking the bitwise logic).
    """
    # 1. Map inputs
    signals = list(input_tuple)
    
    # 2. Simulate gates
    # We must use the same GATE_OPS mapping but for single bits
    for gate in individual:
        gtype = gate['type']
        ins = gate['inputs']
        vals = [signals[i] if i < len(signals) else 0 for i in ins]
        
        # Simple boolean logic for verification
        a = vals[0]
        b = vals[1] if len(vals) > 1 else 0
        c = vals[2] if len(vals) > 2 else 0
        
        res = 0
        if gtype == "AND": res = a & b
        elif gtype == "OR":  res = a | b
        elif gtype == "XOR": res = a ^ b
        elif gtype == "NOT": res = 1 - a
        elif gtype == "NAND": res = 1 - (a & b)
        elif gtype == "NOR":  res = 1 - (a | b)
        elif gtype == "XNOR": res = 1 - (a ^ b)
        # Macros
        elif gtype == "MUX2": res = (a if c == 0 else b) # Note: S is usually 1st arg in engine, check mapping!
        # Engine b_MUX2(s, a, b) -> s is vals[0], a is vals[1], b is vals[2]
        elif gtype == "HALF_SUM": res = a ^ b
        elif gtype == "HALF_CARRY": res = a & b
        elif gtype == "FULL_SUM": res = a ^ b ^ c
        elif gtype == "FULL_CARRY": res = (a & b) | (a & c) | (b & c)
        elif gtype == "EQ1": res = 1 - (a ^ b)
        elif gtype == "GT1": res = 1 if a > b else 0
        
        signals.append(res & 1) # Keep it 0/1
        
    return signals

def print_results(best_ind, num_inputs, num_outputs, inputs, targets):
    print("\n" + "="*40)
    print("       FINAL CIRCUIT ANALYSIS       ")
    print("="*40)
    
    if not best_ind:
        print("❌ No valid circuit evolved.")
        return

    # 1. Gate List
    print(f"\n[Genome Structure] ({len(best_ind)} gates)")
    print(f"{'Idx':<4} | {'Type':<12} | {'Inputs'}")
    print("-" * 30)
    for i, gate in enumerate(best_ind):
        # Convert absolute input indices to readable names
        inputs_str = []
        for inp in gate['inputs']:
            if inp < num_inputs:
                inputs_str.append(f"I{inp}")
            else:
                inputs_str.append(f"g{inp - num_inputs}")
        print(f"g{i:<3} | {gate['type']:<12} | {', '.join(inputs_str)}")

    # 2. Truth Table Check
    print("\n[Truth Table Verification]")
    # Header
    header = " ".join([f"I{i}" for i in range(num_inputs)]) + " | " + \
             " ".join([f"O{i}" for i in range(num_outputs)]) + " | Target"
    print(header)
    print("-" * len(header))
    
    correct_count = 0
    total_count = len(inputs) * num_outputs
    
    # Check the LAST N gates as outputs (Standard Phase 5 convention)
    output_indices = [len(best_ind) - num_outputs + i for i in range(num_outputs)]
    
    # We need absolute indices for the signals list
    # signals = [I0, I1... In, g0, g1... gm]
    # Output 0 is signal at index: num_inputs + (total_gates - num_outputs)
    
    for r, row in enumerate(inputs):
        signals = evaluate_serial(best_ind, row)
        
        # Extract outputs
        # The signals list has length: num_inputs + num_gates
        # The last N values are the outputs
        circuit_outs = signals[-num_outputs:]
        
        target_outs = [targets[o][r] for o in range(num_outputs)]
        
        # Print row
        in_str = "  ".join(map(str, row))
        out_str = "  ".join(map(str, circuit_outs))
        tgt_str = "  ".join(map(str, target_outs))
        
        match = (circuit_outs == target_outs)
        if match: correct_count += len(circuit_outs)
        else: 
            # Calculate partial matches for stats
            for k in range(num_outputs):
                if circuit_outs[k] == target_outs[k]: correct_count += 1
        
        # Only print first 16 rows to avoid spamming console
        if r < 16:
            mark = "✅" if match else "❌"
            print(f"{in_str} | {out_str} | {tgt_str} {mark}")
            
    if len(inputs) > 16:
        print(f"... (skipping {len(inputs)-16} rows) ...")
        
    print(f"\nFinal Accuracy: {correct_count}/{total_count} bits ({correct_count/total_count*100:.1f}%)")


# ==============================================================================
# 2. SCENARIOS
# ==============================================================================

def run_interactive():
    n_in, n_out, inputs, targets = get_user_target()
    
    cfg = engine.Phase5Config(
        num_gates=30,
        pop_size=1000,
        generations_per_bit=1000,
        elitism=20,
        p_choose_primitive=0.6,
        log_every=50
    )
    
    best_ind = engine.evolve_incremental_phase5(n_in, n_out, inputs, targets, cfg)
    print_results(best_ind, n_in, n_out, inputs, targets)

def run_5bit_adder():
    n_in = 10; n_out = 6
    print(f"\n⚡ Generating 5-bit Adder Truth Table ({2**n_in} rows)...")
    
    inputs = []
    targets = [[] for _ in range(n_out)]
    
    for i in range(1024):
        val_a = (i >> 5) & 0x1F
        val_b = i & 0x1F
        inp_bits = tuple((i >> b) & 1 for b in range(9, -1, -1))
        inputs.append(inp_bits)
        total = val_a + val_b
        for bit in range(n_out):
            targets[bit].append((total >> bit) & 1)

    # Configuration tuned for the 5-bit adder challenge
    cfg = engine.Phase5Config(
        num_gates=70,           # Enough space for complex carry chains
        pop_size=2500,          # Large population for diversity
        generations_per_bit=2500, # Patience for hard carry bits (3, 4)
        elitism=50,
        base_mut=0.05,
        min_mut=0.005,
        p_choose_primitive=0.5, # Balance Macros/Primitives
        log_every=10,
        parallel=True
    )
    
    start_time = time.time()
    best_ind = engine.evolve_incremental_phase5(n_in, n_out, inputs, targets, cfg)
    end_time = time.time()
    
    print(f"\n⏱️ Total Execution Time: {end_time - start_time:.2f} seconds")
    print_results(best_ind, n_in, n_out, inputs, targets)

# ==============================================================================
# 3. MAIN
# ==============================================================================

if __name__ == "__main__":
    print("================================================================")
    print("🧬 PHASE 5: HYBRID EVOLUTIONARY ENGINE")
    print("   (Bitwise + Macros + Incremental Locking)")
    print("================================================================\n")
    
    print("1. Interactive Mode")
    print("2. Run 5-bit Adder Benchmark (The Big Test)")
    
    choice = input("\nSelect Option (1-2): ").strip()
    
    if choice == "1":
        run_interactive()
    elif choice == "2":
        run_5bit_adder()
    else:
        print("Invalid selection.")