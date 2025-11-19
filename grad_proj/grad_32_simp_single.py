from sympy import symbols, simplify_logic, Not, And, Or, Xor

# Gate mapping to Sympy
GATE_MAP = {
    "AND": And,
    "OR": Or,
    "XOR": Xor,
    "NAND": lambda a, b: Not(And(a, b)),
    "NOR": lambda a, b: Not(Or(a, b)),
    "XNOR": lambda a, b: Not(Xor(a, b)),
}
def simplify_with_patterns(expr, target_patterns):
    """
    Try to match expr to one of the target patterns (list of sympy exprs).
    Return the matched pattern if equivalent, else SOP form.
    """
    for pattern in target_patterns:
        if expr.equals(pattern):
            return pattern
    return simplify_logic(expr, form="dnf")
def get_symbol(name):
    """Convert GA variable name to Sympy symbol or inverted symbol."""
    if name.startswith("n") and len(name) > 1:
        return Not(symbols(name[1:]))
    return symbols(name)

def build_expr(network):
    """Build expression map for the network."""
    expr_map = {}
    for gate in network:
        in1, in2 = gate["inputs"]
        in1_expr = expr_map.get(in1, get_symbol(in1))
        in2_expr = expr_map.get(in2, get_symbol(in2))
        expr_map[gate["name"]] = GATE_MAP[gate["gate"]](in1_expr, in2_expr)
    return expr_map

def simplify_single_output(network, output_gate_name):
    """
    Simplify a single output from a GA-evolved network.
    - network: list of gates
    - output_gate_name: name of the gate that produces the final output
    """
    expr_map = build_expr(network)
    final_expr = expr_map[output_gate_name]
    return simplify_logic(final_expr, form="dnf")
