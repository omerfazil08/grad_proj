# main_colab_phase6.py
# Driver for Phase 6: Island Model Evolution

import matplotlib.pyplot as plt
import sys
import vhdl_generator_1 
import evolution_colab_phase6 

try:
    from simplifier_phase14 import simplify_genome
    HAS_SIMPLIFIER = True
except ImportError:
    HAS_SIMPLIFIER = False

def run_5bit_adder_islands():
    n_in = 10; n_out = 6
    print(f"Generating 1024-row Truth Table for 5-bit Adder...")
    
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

    # Configuration for Island Model
    cfg = evolution_colab_phase6.BitwiseConfig(
        num_gates=80, 
        pop_size=3000, # Total population (split across islands)
        generations=8000, 
        elitism=10,    # Per island
        tournament_k=5,
        base_mut=0.08, min_mut=0.01,
        p_choose_primitive=0.60,
        log_every=20,
        record_history=False, 
        seed=42, 
        size_penalty_lambda=0.0, 
        parallel=True,
        num_islands=6  # 6 Islands of 500 individuals
    )
    
    print("🚀 Starting Phase 6 (Island Model + Diversity Fix)...")
    
    best_ind_raw, _, _, _ = evolution_colab_phase6.evolve_islands(
        n_in, n_out, inputs, targets, cfg
    )
    
    # Final Export
    if best_ind_raw:
        print("Converting genome for export...")
        best_ind_str = evolution_colab_phase6.convert_to_string_format(best_ind_raw, n_in)
        vhdl = vhdl_generator_1.generate_vhdl_code(best_ind_str, n_in, n_out)
        with open("evolved_5bit_adder_phase6.vhd", "w") as f:
            f.write(vhdl)
        print("✅ VHDL Exported: evolved_5bit_adder_phase6.vhd")
    else:
        print("❌ Evolution failed to produce a valid result.")

if __name__ == "__main__":
    run_5bit_adder_islands()