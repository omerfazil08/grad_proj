# main_colab_phase2.py
# Driver for Incremental Evolution (Curriculum Learning) - Phase 2
# Updated with Gate Printing & 3-bit Adder Option

import matplotlib.pyplot as plt
import sys
import vhdl_generator
# Import core engine
try:
    from evolution_colab_phase23 import (
        ColabConfig,
        evolve_colab_phase2,
        print_results_phase2,
    )
except ImportError:
    print("❌ Error: 'evolution_colab_phase2.py' not found. Please upload it.")
    sys.exit(1)

# Try to import the simplifier, handle if missing
try:
    from simplifier_phase12 import simplify_genome
    HAS_SIMPLIFIER = True
except ImportError:
    print("Warning: simplifier_phase12.py not found. Logic simplification disabled.")
    HAS_SIMPLIFIER = False

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def get_user_target():
    """Interactive input for custom truth tables."""
    try:
        num_inputs = int(input("Enter number of inputs (2..8): ").strip())
        num_outputs = int(input("Enter number of outputs (1..4): ").strip())
    except ValueError:
        print("Invalid input. Defaulting to 2 inputs, 1 output.")
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
        # Safety padding
        if len(vals) != len(rows):
             print(f"Warning: Input length mismatch. Expected {len(rows)}, got {len(vals)}. Padding with 0.")
             vals = vals + [0]*(len(rows)-len(vals))
             vals = vals[:len(rows)]
        targets.append(vals)
        
    return num_inputs, num_outputs, rows, targets

def print_gates_pretty(individual):
    """Prints the raw gate list in a readable format."""
    print("\n[RAW EVOLVED CIRCUIT]")
    print("Gate   | Type       | Inputs")
    print("-------|------------|-------------------")
    for g in individual:
        # Format inputs string
        inputs_str = ", ".join(g['inputs'])
        print(f"{g['name']:6} | {g['type']:10} | {inputs_str}")
    print("-------|------------|-------------------\n")

def run_and_visualize(n_in, n_out, inputs, targets, cfg, title="Evolution Run"):
    """Helper to run evolution, print results, simplify, and plot."""
    print(f"\n🚀 Starting {title}...")
    
    # Run the Phase 2 Engine
    best_ind, best_bkd, hof = evolve_colab_phase2(
        n_in, n_out, inputs, targets, cfg
    )
    
    # Print Stats
    print_results_phase2(best_ind, best_bkd, hof, inputs, targets)
    
    # --- NEW: Print Raw Gates ---
    print_gates_pretty(best_ind)
    
    # Run Simplifier
    if HAS_SIMPLIFIER:
        print("🧠 Simplification of Best Final Circuit:")
        # Generate labels A, B, C... based on input count
        input_names = [chr(ord('A') + i) for i in range(n_in)]
        out_gates = [best_ind[-(n_out-i)]["output"] for i in range(n_out)]
        try:
            res = simplify_genome(best_ind, input_names, out_gates)
            for k, v in res.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  Simplification error: {e}")

# ==============================================================================
# 2. SCENARIOS
# ==============================================================================

def run_interactive():
    n_in, n_out, inputs, targets = get_user_target()
    
    cfg = ColabConfig(
        num_gates=24,
        pop_size=1000,
        generations=1500,
        elitism=15,
        tournament_k=6,
        base_mut=0.05,
        min_mut=0.005,
        p_choose_primitive=0.70,
        log_every=50,
        record_history=True,
        seed=42,
        size_penalty_lambda=0.0,
        parallel=True
    )
    run_and_visualize(n_in, n_out, inputs, targets, cfg, "Interactive Mode")

def run_full_adder_3in_2out():
    # Standard 1-bit Full Adder
    n_in = 3 # A, B, Cin
    n_out = 2 # Sum, Cout
    
    # Inputs (000 to 111)
    inputs = [( (i>>2)&1, (i>>1)&1, (i>>0)&1 ) for i in range(8)]
    
    # Targets
    # Sum = A ^ B ^ Cin
    S = [0, 1, 1, 0, 1, 0, 0, 1]
    # Cout = Maj(A, B, Cin) -> (A&B)|(B&C)|(A&C)
    C = [0, 0, 0, 1, 0, 1, 1, 1]
    
    targets = [S, C]

    cfg = ColabConfig(
        num_gates=16,
        pop_size=1000,
        generations=1000,
        elitism=15,
        tournament_k=6,
        base_mut=0.05,
        min_mut=0.005,
        p_choose_primitive=0.70,
        log_every=20,
        record_history=True,
        seed=42,
        size_penalty_lambda=0.0,
        parallel=True
    )
    run_and_visualize(n_in, n_out, inputs, targets, cfg, "Full Adder (1-bit)")

def run_full_adder_4in_3out():
    # 2-bit Adder (No Carry In)
    # Inputs: A[1:0], B[1:0] -> 4 inputs
    # Outputs: Sum[1:0], Cout -> 3 outputs
    
    print("\n⚠️ 2-bit Adder (4 inputs, 3 outputs)")
    print("Logic: A[1:0] + B[1:0] -> {Cout, Sum1, Sum0}")
    
    n_in = 4
    n_out = 3
    
    # 4 inputs: A1, A0, B1, B0
    # 2^4 = 16 rows
    inputs = []
    targets_s0 = []
    targets_s1 = []
    targets_cout = []
    
    for i in range(16):
        # Decode bits
        a1  = (i >> 3) & 1
        a0  = (i >> 2) & 1
        b1  = (i >> 1) & 1
        b0  = (i >> 0) & 1
        
        inputs.append((a1, a0, b1, b0))
        
        # Math
        val_a = (a1 << 1) | a0
        val_b = (b1 << 1) | b0
        total = val_a + val_b
        
        # Outputs
        targets_s0.append((total >> 0) & 1)
        targets_s1.append((total >> 1) & 1)
        targets_cout.append((total >> 2) & 1)
        
    targets = [targets_s0, targets_s1, targets_cout]
    
    cfg = ColabConfig(
        num_gates=24,   # Needs more gates
        pop_size=1200,  # Larger population for stability
        generations=2000,
        elitism=20,
        tournament_k=6,
        base_mut=0.05, 
        min_mut=0.005,
        p_choose_primitive=0.60, 
        log_every=50,
        record_history=True,
        seed=42,
        size_penalty_lambda=0.0,
        parallel=True
    )
    
    run_and_visualize(n_in, n_out, inputs, targets, cfg, "2-bit Adder (4-in, 3-out)")

def run_full_adder_6in_4out():
    # 3-bit Adder (No Carry In)
    # Inputs: A[2:0], B[2:0] -> 6 inputs
    # Outputs: Sum[2:0], Cout -> 4 outputs
    
    print("\n⚠️ 3-bit Adder (6 inputs, 4 outputs) - EXTREME TEST")
    print("Logic: A[2:0] + B[2:0] -> {Cout, Sum2, Sum1, Sum0}")
    print("Table Size: 64 rows. Search Space: Huge.")
    
    n_in = 6
    n_out = 4
    
    # 6 inputs: A2, A1, A0, B2, B1, B0
    # 2^6 = 64 rows
    inputs = []
    targets_s0 = []
    targets_s1 = []
    targets_s2 = []
    targets_cout = []
    
    for i in range(64):
        # Decode bits
        a2 = (i >> 5) & 1
        a1 = (i >> 4) & 1
        a0 = (i >> 3) & 1
        b2 = (i >> 2) & 1
        b1 = (i >> 1) & 1
        b0 = (i >> 0) & 1
        
        inputs.append((a2, a1, a0, b2, b1, b0))
        
        # Math
        val_a = (a2 << 2) | (a1 << 1) | a0
        val_b = (b2 << 2) | (b1 << 1) | b0
        total = val_a + val_b
        
        # Outputs
        targets_s0.append((total >> 0) & 1)
        targets_s1.append((total >> 1) & 1)
        targets_s2.append((total >> 2) & 1)
        targets_cout.append((total >> 3) & 1)
        
    targets = [targets_s0, targets_s1, targets_s2, targets_cout]
    
    cfg = ColabConfig(
        num_gates=40,   # Significant jump in gates needed
        pop_size=2000,  # Needs very large population
        generations=4000, # Needs time
        elitism=25,
        tournament_k=7,
        base_mut=0.05, 
        min_mut=0.005,
        p_choose_primitive=0.60, # Macros essential here
        log_every=10,
        record_history=True,
        seed=42,
        size_penalty_lambda=0.0,
        parallel=True
    )
    
    run_and_visualize(n_in, n_out, inputs, targets, cfg, "3-bit Adder (6-in, 4-out)")


# ==============================================================================
# 3. MAIN MENU
# ==============================================================================

if __name__ == "__main__":
    print("================================================================")
    print("🧬 INCREMENTAL EVOLUTIONARY LOGIC (Phase 2)")
    print("   Features: Curriculum Learning, Hall of Fame, Multi-Objective")
    print("================================================================\n")
    
    print("1. Interactive Mode (Custom Inputs)")
    print("2. Test: 1-bit Full Adder (3 In -> 2 Out)")
    print("3. Test: 2-bit Adder (4 In -> 3 Out)")
    print("4. Test: 3-bit Adder (6 In -> 4 Out) [EXTREME]")
    
    choice = input("\nSelect Option (1-4): ").strip()
    
    if choice == "1":
        run_interactive()
    elif choice == "2":
        run_full_adder_3in_2out()
    elif choice == "3":
        run_full_adder_4in_3out()
    elif choice == "4":
        run_full_adder_6in_4out()
    else:
        print("Invalid selection.")