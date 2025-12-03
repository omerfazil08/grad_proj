# main_colab_phase4.py
# Phase 4 Driver: Divide & Conquer
# Evolve N independent circuits -> Merge -> VHDL Export

import sys
import vhdl_generator_1 # Ensure this is robust version
import evolution_colab_phase4 # Single-objective engine

try:
    from simplifier_phase14 import simplify_genome
    HAS_SIMPLIFIER = True
except ImportError:
    HAS_SIMPLIFIER = False

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
    for o in range(num_outputs):
        print(f"Generating Truth Table for Output {o+1}...")
        # Placeholder: We usually generate this programmatically for complex adders
        # because typing 1024 bits is impossible.
        # See specific run functions below.
        pass
        
    return num_inputs, num_outputs, rows, targets

def merge_circuits(subcircuits, num_inputs):
    """
    Merges multiple independent gate lists into one large circuit.
    Renames internal gates to avoid collisions (G0, G1 -> G50, G51).
    Retains shared Inputs (A0..An).
    """
    merged_gates = []
    final_output_gates = []
    
    global_gate_offset = 0
    
    print("\n🔗 Merging Sub-Circuits...")
    
    for i, circuit in enumerate(subcircuits):
        # circuit is a list of dicts: [{'name': 'G0', 'inputs': ['A0', 'A1']...}]
        
        # We need to remap its internal gates
        # Input map: 'A0'->'A0', 'G0'->'G(offset+0)'
        
        # Find the name of the last gate (the output of this subcircuit)
        last_local_gate = circuit[-1]['name'] # e.g. "G15"
        
        for gate in circuit:
            # Create new name
            # Assuming gate['name'] format "G<int>"
            local_idx = int(gate['name'][1:])
            new_name = f"G{global_gate_offset + local_idx}"
            
            # Remap inputs
            new_inputs = []
            for inp in gate['inputs']:
                if inp.startswith("A"): # Primary input
                    new_inputs.append(inp)
                elif inp.startswith("G"): # Internal wire
                    loc_inp_idx = int(inp[1:])
                    new_inputs.append(f"G{global_gate_offset + loc_inp_idx}")
            
            merged_gates.append({
                'name': new_name,
                'type': gate['type'],
                'inputs': new_inputs,
                'output': new_name
            })
            
        # Register the output gate for this subcircuit
        # The last gate added to merged_gates corresponds to the output of this subcircuit
        final_output_gates.append(merged_gates[-1]['name'])
        
        global_gate_offset += len(circuit)
        
    print(f"   Merged Total: {len(merged_gates)} gates.")
    return merged_gates, final_output_gates

def run_5bit_divide_conquer():
    n_in = 10; n_out = 6
    print(f"Generating 1024-row Truth Table for 5-bit Adder...")
    
    inputs = []
    # Prepare targets columns
    targets = [[] for _ in range(n_out)]
    
    for i in range(1024):
        val_a = (i >> 5) & 0x1F
        val_b = i & 0x1F
        inp_bits = tuple((i >> b) & 1 for b in range(9, -1, -1))
        inputs.append(inp_bits)
        total = val_a + val_b
        for bit in range(n_out):
            targets[bit].append((total >> bit) & 1)

    # Config for SINGLE OUTPUT (Smaller/Faster)
    # Since we solve 1 bit at a time, we don't need 60 gates. 
    # 15-20 gates is enough for one Full Adder bit logic.
    cfg = evolution_colab_phase4.Phase4Config(
        num_gates=20,           # Much smaller search space!
        pop_size=1000, 
        generations=2000, 
        elitism=10,
        tournament_k=5,
        base_mut=0.05, min_mut=0.005, 
        p_choose_primitive=0.70,
        log_every=100,
        record_history=False, 
        seed=42, 
        parallel=True
    )
    
    subcircuits = []
    
    # Run Evolution Loop
    for i in range(n_out):
        print(f"\n--- Evolving Output {i+1}/{n_out} ---")
        # Increase gates slightly for higher bits (more carry logic needed)
        if i > 2: cfg.num_gates = 30 
        if i > 4: cfg.num_gates = 40
        
        best_raw = evolution_colab_phase4.evolve_single_output(
            n_in, inputs, targets[i], cfg, label=f"Bit_{i}"
        )
        
        # Convert to string format
        best_str = evolution_colab_phase4.convert_single_to_string(best_raw, n_in)
        
        # Simplify BEFORE merging (Optional but recommended)
        if HAS_SIMPLIFIER:
            # We just simplify the logic of the LAST gate (the output)
            print(f"   Simplifying Output {i+1}...")
            # Note: Simplifying and re-synthesizing here would be complex.
            # We will just store the raw evolved circuit for safety in Phase 4.
            pass
            
        subcircuits.append(best_str)

    # Merge
    merged_gates, output_map = merge_circuits(subcircuits, n_in)
    
    # VHDL Export
    # We need to tell VHDL generator that O0 is connected to 'G25', O1 to 'G50' etc.
    # The current vhdl_generator connects O0..On to the *last N gates*.
    # We need to modify the merged list so that the "Output Gates" are physically at the end?
    # OR, better: Use a VHDL generator that accepts an output map.
    
    # Quick Hack: Add buffer gates (OR x, x) at the very end of the list to map specific signals to outputs.
    print("   Wiring final outputs...")
    for i, source_gate in enumerate(output_map):
        # Create a buffer gate: G_final = G_source OR G_source
        new_g = {
            'name': f"G_OUT_{i}",
            'type': 'OR',
            'inputs': [source_gate, source_gate],
            'output': f"G_OUT_{i}"
        }
        merged_gates.append(new_g)
        
    # Now the last N gates correspond exactly to O0..On
    vhdl = vhdl_generator_1.generate_vhdl_code(merged_gates, n_in, n_out)
    
    with open("evolved_5bit_adder_split.vhd", "w") as f:
        f.write(vhdl)
    print("✅ Full System VHDL Exported: evolved_5bit_adder_split.vhd")

if __name__ == "__main__":
    run_5bit_divide_conquer()