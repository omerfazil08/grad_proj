# main_colab_phase71.py
import sys
import vhdl_generator_1 
import evolution_colab_phase71

try:
    from simplifier_phase14 import simplify_genome
    HAS_SIMPLIFIER = True
except ImportError:
    HAS_SIMPLIFIER = False

def run_5bit_adder_phase71():
    n_in = 10; n_out = 6
    print(f"Generating 1024-row Truth Table for 5-bit Adder...")
    
    inputs = []; targets = [[] for _ in range(n_out)]
    for i in range(1024):
        val_a = (i >> 5) & 0x1F
        val_b = i & 0x1F
        inp_bits = tuple((i >> b) & 1 for b in range(9, -1, -1))
        inputs.append(inp_bits)
        total = val_a + val_b
        for bit in range(n_out):
            targets[bit].append((total >> bit) & 1)

    cfg = evolution_colab_phase71.BitwiseConfig(
        num_gates=80, 
        pop_size=3000, 
        generations=15000, 
        elitism=10,
        tournament_k=5,
        base_mut=0.08, min_mut=0.01,
        p_choose_primitive=0.60,
        log_every=20,
        record_history=False, 
        seed=42, 
        size_penalty_lambda=0.0, 
        parallel=True,
        num_islands=6 
    )
    
    print("🚀 Starting Phase 7.1 (Hybrid Trigger + HSS Reset)...")
    print("   > Novelty Threshold: 800 gens")
    print("   > Hybrid Veto: Enabled (0.5% avg improvement)")
    
    best_ind_raw = evolution_colab_phase71.evolve_phase71(
        n_in, n_out, inputs, targets, cfg
    )
    
    if best_ind_raw:
        print("Converting genome for export...")
        best_ind_str = evolution_colab_phase71.convert_to_string_format(best_ind_raw, n_in)
        vhdl = vhdl_generator_1.generate_vhdl_code(best_ind_str, n_in, n_out)
        with open("evolved_5bit_adder_phase71.vhd", "w") as f:
            f.write(vhdl)
        print("✅ VHDL Exported.")

if __name__ == "__main__":
    run_5bit_adder_phase71()