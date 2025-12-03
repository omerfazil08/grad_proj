# main_colab_phase25.py
# Phase 2.3 Driver: Evolution + VHDL Export + Visualization
# Updated to fix VHDL export and plotting

import matplotlib.pyplot as plt
import numpy as np
import sys
import vhdl_generator  # Ensure vhdl_generator.py is in the same folder

# Import Engine
try:
    from evolution_colab_phase23 import (
        ColabConfig,
        evolve_colab_phase2,
        print_results_phase2,
    )
except ImportError:
    print("❌ Error: 'evolution_colab_phase23.py' missing.")
    sys.exit(1)

# Import Simplifier
try:
    from simplifier_phase12 import simplify_genome
    HAS_SIMPLIFIER = True
except ImportError:
    print("Warning: Simplifier missing.")
    HAS_SIMPLIFIER = False

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def get_user_target():
    try:
        num_inputs = int(input("Enter number of inputs (2..10): ").strip())
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
        val_str = input(f"Enter output truth table #{o+1}:\n→ ").strip()
        vals = [int(v) for v in val_str.split()]
        if len(vals) < len(rows):
             vals += [0] * (len(rows) - len(vals))
        targets.append(vals[:len(rows)])
        
    return num_inputs, num_outputs, rows, targets

def print_gates_pretty(individual):
    print("\n[RAW EVOLVED CIRCUIT]")
    print(f"{'Gate':<6} | {'Type':<10} | {'Inputs'}")
    print("-" * 35)
    for g in individual:
        inputs_str = ", ".join(g['inputs'])
        print(f"{g['name']:<6} | {g['type']:<10} | {inputs_str}")
    print("-" * 35)

def plot_search_gradient(history, filename="search_gradient.png"):
    if not history or 'mu' not in history: return

    generations = history['gen']
    best_fit = np.array(history['best'])
    mu = np.array(history['mu'])
    sigma = np.array(history['sigma'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- PLOT 1: Fitness ---
    ax1.set_title("Evolutionary Progress: The Search Tube", fontsize=12)
    ax1.plot(generations, best_fit, color='green', label='Best Fitness', linewidth=2)
    ax1.plot(generations, mu, color='blue', label=r'Average Fitness ($\mu$)', linestyle='--')
    
    lower_bound = mu - sigma
    upper_bound = mu + sigma
    ax1.fill_between(generations, lower_bound, upper_bound, color='blue', alpha=0.2, label=r'Diversity ($\mu \pm \sigma$)')
    ax1.set_ylabel("Fitness")
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- PLOT 2: Gradient (Sigma) ---
    ax2.set_title(r"Search Gradient (Population Diversity $\sigma$)", fontsize=12)
    ax2.plot(generations, sigma, color='orange', label=r'Std Dev ($\sigma$)', linewidth=2)
    ax2.set_ylabel("Sigma")
    ax2.set_xlabel("Generations")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n📊 Plot saved to {filename} (Check Files tab)")
    plt.show()

def run_and_visualize(n_in, n_out, inputs, targets, cfg, title="Evolution Run"):
    print(f"\n🚀 Starting {title}...")
    
    # 1. Run Evolution
    best_ind, best_bkd, hof, history = evolve_colab_phase2(
        n_in, n_out, inputs, targets, cfg
    )
    
    # 2. Print Results
    print_results_phase2(best_ind, best_bkd, hof, inputs, targets)
    print_gates_pretty(best_ind)
    
    # 3. Simplify
    if HAS_SIMPLIFIER:
        print("\n🧠 Simplification of Best Final Circuit:")
        input_names = [chr(ord('A') + i) for i in range(n_in)]
        out_gates = [best_ind[-(n_out-i)]["output"] for i in range(n_out)]
        try:
            res = simplify_genome(best_ind, input_names, out_gates)
            for k, v in res.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  Simplification error: {e}")

    # 4. VHDL Export
    print("\n💾 Generating VHDL...")
    try:
        # Use the imported module explicitly
        vhdl_code = vhdl_generator.generate_vhdl_code(best_ind, n_in, n_out)
        clean_title = title.lower().replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"evolved_{clean_title}.vhd"
        with open(filename, "w") as f:
            f.write(vhdl_code)
        print(f"   ✅ VHDL saved to: {filename}")
    except Exception as e:
        print(f"   ❌ VHDL Generation Failed: {e}")

    # 5. Plot
    plot_search_gradient(history)

# ==============================================================================
# 2. PRE-DEFINED SCENARIOS
# ==============================================================================

def run_interactive():
    n_in, n_out, inputs, targets = get_user_target()
    cfg = ColabConfig(
        num_gates=20, pop_size=500, generations=1000, elitism=10, tournament_k=5,
        base_mut=0.10, min_mut=0.01, p_choose_primitive=0.70, log_every=50,
        record_history=True, seed=42, size_penalty_lambda=0.0, parallel=True
    )
    run_and_visualize(n_in, n_out, inputs, targets, cfg, "Interactive Mode")

def run_1bit_adder():
    n_in = 3; n_out = 2
    inputs = [( (i>>2)&1, (i>>1)&1, (i>>0)&1 ) for i in range(8)]
    S = [0, 1, 1, 0, 1, 0, 0, 1]
    C = [0, 0, 0, 1, 0, 1, 1, 1]
    cfg = ColabConfig(
        num_gates=16, pop_size=800, generations=1000, elitism=15, tournament_k=6,
        base_mut=0.05, min_mut=0.005, p_choose_primitive=0.70, log_every=20,
        record_history=True, seed=42, size_penalty_lambda=0.0, parallel=True
    )
    run_and_visualize(n_in, n_out, inputs, [S, C], cfg, "1-bit Adder")

def run_2bit_adder():
    n_in = 4; n_out = 3
    inputs = []; targets = [[], [], []]
    for i in range(16):
        a1=(i>>3)&1; a0=(i>>2)&1; b1=(i>>1)&1; b0=(i>>0)&1
        inputs.append((a1, a0, b1, b0))
        val = ((a1<<1)|a0) + ((b1<<1)|b0)
        targets[0].append((val>>0)&1); targets[1].append((val>>1)&1); targets[2].append((val>>2)&1)
    cfg = ColabConfig(
        num_gates=24, pop_size=1000, generations=2000, elitism=20, tournament_k=6,
        base_mut=0.05, min_mut=0.005, p_choose_primitive=0.60, log_every=50,
        record_history=True, seed=42, size_penalty_lambda=0.0, parallel=True
    )
    run_and_visualize(n_in, n_out, inputs, targets, cfg, "2-bit Adder")

def run_3bit_adder():
    n_in = 6; n_out = 4
    inputs = []; targets = [[], [], [], []]
    for i in range(64):
        a2=(i>>5)&1; a1=(i>>4)&1; a0=(i>>3)&1
        b2=(i>>2)&1; b1=(i>>1)&1; b0=(i>>0)&1
        inputs.append((a2,a1,a0,b2,b1,b0))
        val = ((a2<<2)|(a1<<1)|a0) + ((b2<<2)|(b1<<1)|b0)
        for bit in range(4):
            targets[bit].append((val>>bit)&1)

    cfg = ColabConfig(
        num_gates=40, pop_size=2000, generations=4000, elitism=25, tournament_k=7,
        base_mut=0.05, min_mut=0.005, p_choose_primitive=0.60, log_every=50,
        record_history=True, seed=42, size_penalty_lambda=0.0, parallel=True
    )
    run_and_visualize(n_in, n_out, inputs, targets, cfg, "3-bit Adder")

def run_5bit_adder_complex():
    n_in = 10; n_out = 6
    print(f"Generating Truth Table for {n_in} inputs (1024 rows)...")
    inputs = []; targets = [[] for _ in range(n_out)]
    for i in range(1024):
        val_a = (i >> 5) & 0x1F
        val_b = i & 0x1F
        inp_bits = tuple((i >> b) & 1 for b in range(9, -1, -1))
        inputs.append(inp_bits)
        total = val_a + val_b
        for bit in range(n_out):
            targets[bit].append((total >> bit) & 1)

    cfg = ColabConfig(
        num_gates=60, pop_size=2500, generations=6000, elitism=30, tournament_k=8,
        base_mut=0.05, min_mut=0.005, p_choose_primitive=0.60,
        log_every=50, record_history=True, seed=42, size_penalty_lambda=0.0, parallel=True
    )
    run_and_visualize(n_in, n_out, inputs, targets, cfg, "5-bit Adder")

if __name__ == "__main__":
    print("================================================================")
    print("🧬 EVOLUTIONARY LOGIC: Phase 2.3")
    print("================================================================\n")
    print("1. Interactive Mode")
    print("2. 1-bit Full Adder (3 In -> 2 Out)")
    print("3. 2-bit Full Adder (4 In -> 3 Out)")
    print("4. 3-bit Full Adder (6 In -> 4 Out)")
    print("5. 5-bit Full Adder (10 In -> 6 Out) [Complex Benchmark]")
    
    choice = input("\nSelect Option (1-5): ").strip()
    
    if choice == "1": run_interactive()
    elif choice == "2": run_1bit_adder()
    elif choice == "3": run_2bit_adder()
    elif choice == "4": run_3bit_adder()
    elif choice == "5": run_5bit_adder_complex()
    else: print("Invalid selection.")