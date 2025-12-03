# main_colab_phase24.py
# Phase 2.3 Driver: Evolution + VHDL Export + Visualization
# FIXED: VHDL Generation types and Plotting syntax

import matplotlib.pyplot as plt
import numpy as np
import sys
import vhdl_generator  # <--- Using the NEW vhdl_generator

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

# --- Helper Functions ---

def get_user_target():
    try:
        num_inputs = int(input("Enter number of inputs (2..8): ").strip())
        num_outputs = int(input("Enter number of outputs (1..4): ").strip())
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
        targets.append(vals)
        
    return num_inputs, num_outputs, rows, targets

def plot_search_gradient(history):
    if not history or 'mu' not in history: return

    generations = history['gen']
    best_fit = np.array(history['best'])
    mu = np.array(history['mu'])
    sigma = np.array(history['sigma'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Fixed raw strings (r'') to prevent SyntaxWarning
    ax1.set_title("Evolutionary Progress: The Search Tube", fontsize=12)
    ax1.plot(generations, best_fit, color='green', label='Best Fitness', linewidth=2)
    ax1.plot(generations, mu, color='blue', label=r'Average Fitness ($\mu$)', linestyle='--')
    
    lower_bound = mu - sigma
    upper_bound = mu + sigma
    ax1.fill_between(generations, lower_bound, upper_bound, color='blue', alpha=0.2, label=r'Diversity ($\mu \pm \sigma$)')
    ax1.set_ylabel("Fitness")
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.set_title(r"Search Gradient (Population Diversity $\sigma$)", fontsize=12)
    ax2.plot(generations, sigma, color='orange', label=r'Std Dev ($\sigma$)', linewidth=2)
    ax2.set_ylabel("Sigma")
    ax2.set_xlabel("Generations")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def run_and_visualize(n_in, n_out, inputs, targets, cfg, title="Evolution Run"):
    print(f"\n🚀 Starting {title}...")
    
    # 1. Run Evolution
    best_ind, best_bkd, hof, history = evolve_colab_phase2(
        n_in, n_out, inputs, targets, cfg
    )
    
    # 2. Print Results
    print_results_phase2(best_ind, best_bkd, hof, inputs, targets)
    
    # 3. VHDL Export (FIXED CALL)
    print("\n💾 Generating VHDL...")
    try:
        # Pass the list 'best_ind' directly to the new generator
        vhdl_code = vhdl_generator.generate_vhdl_code(best_ind, n_in, n_out)
        filename = f"evolved_{title.lower().replace(' ', '_')}.vhd"
        with open(filename, "w") as f:
            f.write(vhdl_code)
        print(f"   ✅ VHDL saved to: {filename}")
        print("   (Snippet):")
        print("\n".join(vhdl_code.splitlines()[:15]) + "\n   ...")
    except Exception as e:
        print(f"   ❌ VHDL Generation Failed: {e}")

    # 4. Plot
    print("\n📊 Plotting Gradient...")
    plot_search_gradient(history)

# --- Scenarios ---

def run_full_adder_3in_2out():
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

def run_2bit_adder_4in_3out():
    n_in = 4; n_out = 3
    inputs = []; t_s0=[]; t_s1=[]; t_cout=[]
    for i in range(16):
        a1=(i>>3)&1; a0=(i>>2)&1; b1=(i>>1)&1; b0=(i>>0)&1
        inputs.append((a1, a0, b1, b0))
        val = ((a1<<1)|a0) + ((b1<<1)|b0)
        t_s0.append((val>>0)&1); t_s1.append((val>>1)&1); t_cout.append((val>>2)&1)
        
    cfg = ColabConfig(
        num_gates=24, pop_size=1000, generations=2000, elitism=20, tournament_k=6,
        base_mut=0.05, min_mut=0.005, p_choose_primitive=0.60, log_every=50,
        record_history=True, seed=42, size_penalty_lambda=0.0, parallel=True
    )
    run_and_visualize(n_in, n_out, inputs, [t_s0, t_s1, t_cout], cfg, "2-bit Adder")

if __name__ == "__main__":
    print("1. 1-bit Adder")
    print("2. 2-bit Adder")
    c = input("Choice: ")
    if c == "1": run_full_adder_3in_2out()
    elif c == "2": run_2bit_adder_4in_3out()