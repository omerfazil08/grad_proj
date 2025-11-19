from sympy import symbols, simplify_logic, Not, And, Or, Xor
from sympy.logic.boolalg import boolalg
# Base symbols
A, B, C, D, EN, Q = symbols("A B C D EN Q")

# Gate mapping to sympy
GATE_MAP = {
    "AND": And,
    "OR": Or,
    "XOR": Xor,
    "NAND": lambda a, b: Not(And(a, b)),
    "NOR": lambda a, b: Not(Or(a, b)),
    "XNOR": lambda a, b: Not(Xor(a, b)),
}

def get_symbol(name):
    """Convert GA variable name to sympy symbol or inverted symbol."""
    # Handle inverted inputs
    if name.startswith("n") and len(name) > 1:
        base_name = name[1:]
        return Not(symbols(base_name))
    return symbols(name)

def build_expr(gate, expr_map):
    """Recursively resolve gate inputs into Sympy expressions."""
    in1, in2 = gate["inputs"]

    in1_expr = expr_map.get(in1, get_symbol(in1))
    in2_expr = expr_map.get(in2, get_symbol(in2))

    return GATE_MAP[gate["gate"]](in1_expr, in2_expr)

def simplify_network(network, component_type="generic"):
    """Simplify GA-evolved network into clean boolean formulas."""
    expr_map = {}

    # Build expressions for each gate in order
    for g in network:
        expr_map[g["name"]] = build_expr(g, expr_map)

    if component_type == "full_adder":
        sum_expr = expr_map[network[-2]["name"]]
        carry_expr = expr_map[network[-1]["name"]]
        sum_pretty = boolalg.Equivalent(sum_expr, Xor(A, B, C))
        return {
            "SUM": simplify_logic(sum_pretty, form="dnf"),
            "CARRY": simplify_logic(carry_expr, form="dnf")
        }

    elif component_type in ["d_latch", "d_flip_flop"]:
        q_expr = expr_map[network[-2]["name"]]
        nq_expr = expr_map[network[-1]["name"]]
        return {
            "Q": simplify_logic(q_expr, form="dnf"),
            "nQ": simplify_logic(nq_expr, form="dnf")
        }

    else:  # generic
        out_expr = expr_map[network[-1]["name"]]
        return {"OUT": simplify_logic(out_expr, form="dnf")}