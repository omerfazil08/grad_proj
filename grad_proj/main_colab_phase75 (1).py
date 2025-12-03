# main_colab_phase75.py
import sys
import vhdl_generator_1 
import evolution_colab_phase75

try:
    from simplifier_phase74 import simplify_genome
    HAS_SIMPLIFIER = True
except ImportError:
    print("⚠️  simplifier_phase74.py missing. Compression will be skipped.")
    HAS_SIMPLIFIER = False

def run_5bit_adder_phase75():
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

    cfg = evolution_colab_phase75.BitwiseConfig(
        gate_growth_buffer=30, # 30 Free gates for search
        pop_size=3000, 
        generations=20000, 
        elitism=10,
        tournament_k=5,
        base_mut=0.10, min_mut=0.02, # Higher mutation for smaller genome
        p_choose_primitive=0.60,
        log_every=20,
        record_history=False, 
        seed=42, 
        size_penalty_lambda=0.0, 
        parallel=True,
        num_islands=6 
    )
    
    print("🚀 Starting Phase 7.5 (Dynamic Rolling Buffer)...")
    
    best_ind_raw = evolution_colab_phase75.evolve_phase75(
        n_in, n_out, inputs, targets, cfg
    )
    
    if best_ind_raw:
        print("Converting genome for export...")
        # Phase 7.5 doesn't have a fixed num_gates, but the converter just needs the list
        # Note: We need to pass the final inputs logic
        # The old converter function works fine on dynamic lists
        best_ind_str = evolution_colab_phase75.convert_to_string_format(best_ind_raw, n_in)
        vhdl = vhdl_generator_1.generate_vhdl_code(best_ind_str, n_in, n_out)
        with open("evolved_5bit_adder_phase75.vhd", "w") as f:
            f.write(vhdl)
        print("✅ VHDL Exported.")

if __name__ == "__main__":
    run_5bit_adder_phase75()