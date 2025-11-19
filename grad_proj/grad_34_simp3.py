from sympy import symbols, simplify_logic, Not, And, Or, Xor, Equivalent, simplify

# ==============================================================
# Gate mapping (extended to handle variable-arity gates)
# ==============================================================

def AND_multi(*args): 
    return And(*args)

def OR_multi(*args): 
    return Or(*args)

def XOR_multi(*args): 
    return Xor(*args)

def NAND_multi(*args): 
    return Not(And(*args))

def NOR_multi(*args): 
    return Not(Or(*args))

def XNOR_multi(*args): 
    return Not(Xor(*args))

# Map gate names to corresponding Sympy constructors
GATE_MAP = {
    "AND": AND_multi,
    "OR": OR_multi,
    "XOR": XOR_multi,
    "NAND": NAND_multi,
    "NOR": NOR_multi,
    "XNOR": XNOR_multi,
}

# ==============================================================
# Symbol & simplification utilities
# ==============================================================

# Define default base variables for pattern recognition
A, B, C = symbols("A B C")

TARGET_PATTERNS = [
    Xor(A, B),
    Xor(A, B, C),
    Not(Xor(A, B)),
    Not(Xor(A, B, C))
]

def equivalent(expr1, expr2):
    """Check if two expressions are logically equivalent."""
    try:
        return simplify(expr1 ^ expr2) == False  # expr1 XOR expr2 == False → equivalent
    except Exception:
        return False

def simplify_with_patterns(expr, target_patterns):
    """
    Try to match expr to one of the target patterns (list of sympy exprs).
    Return the matched pattern if equivalent, else SOP form.
    """
    for pattern in target_patterns:
        if equivalent(expr, pattern):
            return pattern
    return simplify_logic(expr, form="dnf")

def get_symbol(name):
    """Convert GA variable name to Sympy symbol or inverted symbol."""
    if name.startswith("n") and len(name) > 1:
        return Not(symbols(name[1:]))
    return symbols(name)

# ==============================================================
# Expression builder (supports variable number of inputs)
# ==============================================================

def build_expr(network):
    """Build Sympy expression map for the evolved network (supports variable-arity gates)."""
    expr_map = {}

    for gate in network:
        inputs = gate.get("inputs", [])
        gate_type = gate.get("gate", "AND")

        # Get expressions for all inputs, building recursively as needed
        input_exprs = []
        for inp in inputs:
            if inp in expr_map:
                input_exprs.append(expr_map[inp])
            else:
                input_exprs.append(get_symbol(inp))

        # Apply the appropriate gate function
        func = GATE_MAP.get(gate_type, AND_multi)
        try:
            expr_map[gate["name"]] = func(*input_exprs)
        except Exception:
            # fallback in case of unexpected input count or type
            expr_map[gate["name"]] = get_symbol(inputs[0]) if inputs else symbols("X")

    return expr_map

# ==============================================================
# Output simplification
# ==============================================================

def simplify_single_output(network, output_gate_name):
    """
    Simplify a single output from a GA-evolved network.
    - network: list of gates (each gate may have 2+ inputs)
    - output_gate_name: name of the gate that produces the final output
    """
    expr_map = build_expr(network)

    if output_gate_name not in expr_map:
        raise ValueError(f"Output gate '{output_gate_name}' not found in network.")

    final_expr = expr_map[output_gate_name]
    return simplify_with_patterns(final_expr, TARGET_PATTERNS)
