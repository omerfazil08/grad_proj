# vhdl_generator.py
# Converts the evolved Python circuit genome into synthesizable VHDL.
# Compatible with Phase 2/3 gate-level representations.

def generate_vhdl_code(individual, num_inputs, num_outputs, entity_name="evolved_circuit"):
    """
    Generates a string containing the full VHDL code for the given individual.
    
    Args:
        individual (object): The evolved circuit object (must have .gates list).
        num_inputs (int): Number of input ports (e.g., 3 for A, B, Cin).
        num_outputs (int): Number of output ports.
        entity_name (str): Name of the VHDL entity.
    
    Returns:
        str: Complete VHDL code.
    """
    
    # 1. Header & Entity
    # ------------------
    vhdl = []
    vhdl.append("library IEEE;")
    vhdl.append("use IEEE.STD_LOGIC_1164.ALL;")
    vhdl.append("")
    vhdl.append(f"entity {entity_name} is")
    vhdl.append("    Port (")
    
    # Inputs: "I0, I1, I2..."
    port_list = []
    for i in range(num_inputs):
        port_list.append(f"        I{i} : in  STD_LOGIC")
        
    # Outputs: "O0, O1..."
    for i in range(num_outputs):
        port_list.append(f"        O{i} : out STD_LOGIC")
        
    # Join ports with semicolons
    vhdl.append(";\n".join(port_list))
    vhdl.append("    );")
    vhdl.append(f"end {entity_name};")
    vhdl.append("")
    
    # 2. Architecture
    # ---------------
    vhdl.append(f"architecture Structural of {entity_name} is")
    
    # Internal Signals (for every gate output)
    # We name signals g0, g1, g2... matching the gate index in the list.
    if len(individual.gates) > 0:
        sig_names = [f"g{i}" for i in range(len(individual.gates))]
        vhdl.append(f"    signal {', '.join(sig_names)} : STD_LOGIC;")
    
    vhdl.append("begin")
    
    # 3. Gate Logic Assignments
    # -------------------------
    # We must map the 'input indices' from the Python object to VHDL signal names.
    # Python Input Index 0 -> VHDL "I0"
    # Python Input Index N -> VHDL "g(N - num_inputs)" (Previous gates)
    
    def get_signal_name(idx):
        if idx < num_inputs:
            return f"I{idx}"  # It's a primary input
        else:
            return f"g{idx - num_inputs}" # It's a gate output
            
    for i, gate in enumerate(individual.gates):
        # Resolve input names
        in1 = get_signal_name(gate.input1)
        # Handle NOT gate (single input) vs others (two inputs)
        if gate.gate_type == 'NOT':
            expr = f"not {in1}"
        else:
            in2 = get_signal_name(gate.input2)
            # Map Python types to VHDL operators
            op = gate.gate_type.lower()
            if op == 'nand':
                expr = f"not ({in1} and {in2})"
            elif op == 'nor':
                expr = f"not ({in1} or {in2})"
            else:
                # AND, OR, XOR
                expr = f"{in1} {op} {in2}"
                
        vhdl.append(f"    g{i} <= {expr};")
        
    # 4. Output Wiring
    # ----------------
    # Connect the LAST N gates to the Output ports (as per Phase 2 convention)
    # or use specific mapping if your individual object has .output_map
    
    vhdl.append("")
    vhdl.append("    -- Output Connections")
    
    start_output_idx = len(individual.gates) - num_outputs
    
    for i in range(num_outputs):
        # Default: Connect last gates to outputs
        gate_idx = start_output_idx + i
        if gate_idx < 0: # Fallback if circuit is too small
             vhdl.append(f"    O{i} <= '0';")
        else:
             vhdl.append(f"    O{i} <= g{gate_idx};")

    vhdl.append("end Structural;")
    
    return "\n".join(vhdl)

# Optional: Self-test block
if __name__ == "__main__":
    # Mock classes for testing
    class MockGate:
        def __init__(self, t, i1, i2=None):
            self.gate_type = t
            self.input1 = i1
            self.input2 = i2
            
    class MockIndividual:
        def __init__(self):
            self.gates = [
                MockGate('XOR', 0, 1), # g0 = I0 xor I1
                MockGate('AND', 0, 1), # g1 = I0 and I1
                MockGate('XOR', 2, 2)  # g2 = g0 xor I2 (Full Adder Sum)
            ]
            
    test_ind = MockIndividual()
    print(generate_vhdl_code(test_ind, num_inputs=3, num_outputs=2))