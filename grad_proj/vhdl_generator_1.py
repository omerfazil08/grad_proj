# vhdl_generator.py
# Robust VHDL Exporter for Phase 2/3 Evolved Circuits
# Handles: Lists of dicts, String inputs ("A0", "G5"), and Macros.

import re

def generate_vhdl_code(individual, num_inputs, num_outputs, entity_name="evolved_circuit"):
    """
    Generates synthesizable VHDL from a list of gate dictionaries.
    """
    # 1. Extract Gate List (Handle list vs object)
    if isinstance(individual, list):
        gates = individual
    elif hasattr(individual, 'gates'):
        gates = individual.gates
    else:
        raise ValueError("Invalid individual format. Expected list of gates.")

    # 2. Header
    lines = []
    lines.append("library IEEE;")
    lines.append("use IEEE.STD_LOGIC_1164.ALL;")
    lines.append("")
    lines.append(f"entity {entity_name} is")
    lines.append("    Port (")
    
    # Define Ports
    port_strs = []
    for i in range(num_inputs):
        port_strs.append(f"        I{i} : in  STD_LOGIC")
    for i in range(num_outputs):
        port_strs.append(f"        O{i} : out STD_LOGIC")
    lines.append(";\n".join(port_strs))
    lines.append("    );")
    lines.append(f"end {entity_name};")
    lines.append("")
    
    # 3. Architecture
    lines.append(f"architecture Structural of {entity_name} is")
    
    # Internal Signals (w_g0, w_g1...)
    if gates:
        sigs = [f"w_{g['name']}" for g in gates if 'name' in g]
        if sigs:
            lines.append(f"    signal {', '.join(sigs)} : STD_LOGIC;")
    
    lines.append("begin")
    
    # 4. Helper to map "A0" -> "I0" and "G5" -> "w_G5"
    def map_sig(s):
        s = str(s)
        if s.startswith("A") and s[1:].isdigit():
            return f"I{s[1:]}" # A0 -> I0
        elif s.startswith("G") and s[1:].isdigit():
            return f"w_{s}"    # G0 -> w_G0
        elif s == "0": return "'0'"
        elif s == "1": return "'1'"
        return s # Fallback

    # 5. Logic Generation
    for gate in gates:
        gname = gate['name']
        gtype = gate['type'].upper()
        inputs = [map_sig(i) for i in gate['inputs']]
        
        expr = ""
        
        # --- Primitive Gates ---
        if gtype == 'NOT':
            expr = f"not {inputs[0]}"
        elif gtype == 'AND':
            expr = " and ".join(inputs)
        elif gtype == 'OR':
            expr = " or ".join(inputs)
        elif gtype == 'XOR' or gtype == 'XOR2':
            expr = f"{inputs[0]} xor {inputs[1]}"
        elif gtype == 'NAND':
            expr = f"not ({' and '.join(inputs)})"
        elif gtype == 'NOR':
            expr = f"not ({' or '.join(inputs)})"
        elif gtype == 'XNOR' or gtype == 'XNOR2':
            expr = f"not ({inputs[0]} xor {inputs[1]})"
            
        # --- Macros (Expansion) ---
        elif gtype == 'MUX2':
            # MUX2(sel, a, b) -> if sel=0 then a else b
            if len(inputs) >= 3:
                s, a, b = inputs[0], inputs[1], inputs[2]
                expr = f"(not {s} and {a}) or ({s} and {b})"
            else:
                expr = "'0'"
                
        elif gtype == 'HALF_SUM':
            expr = f"{inputs[0]} xor {inputs[1]}"
        elif gtype == 'HALF_CARRY':
            expr = f"{inputs[0]} and {inputs[1]}"
        elif gtype == 'FULL_SUM':
            expr = f"{inputs[0]} xor {inputs[1]} xor {inputs[2]}"
        elif gtype == 'FULL_CARRY':
            a, b, c = inputs[0], inputs[1], inputs[2]
            expr = f"({a} and {b}) or ({a} and {c}) or ({b} and {c})"
        elif gtype == 'EQ1':
            expr = f"not ({inputs[0]} xor {inputs[1]})"
            
        else:
            expr = "'0' -- Unknown Gate"

        lines.append(f"    w_{gname} <= {expr};")

    # 6. Output Wiring
    lines.append("")
    lines.append("    -- Outputs")
    start_idx = len(gates) - num_outputs
    for i in range(num_outputs):
        if start_idx + i >= 0:
            target_gate = gates[start_idx + i]['name']
            lines.append(f"    O{i} <= w_{target_gate};")
        else:
            lines.append(f"    O{i} <= '0';")

    lines.append("end Structural;")
    
    return "\n".join(lines)