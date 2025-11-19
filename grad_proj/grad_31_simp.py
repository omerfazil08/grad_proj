from sympy import symbols, simplify_logic, Not, And, Or, Xor

A, B, C, D, EN, Q = symbols("A B C D EN Q")

def simplify_with_patterns(expr, target_patterns):
    """
    Try to match expr to one of the target patterns (list of sympy exprs).
    Return the matched pattern if equivalent, else SOP form.
    """
    for pattern in target_patterns:
        if expr.equals(pattern):
            return pattern
    return simplify_logic(expr, form="dnf")

def simplify_network(network, component_type="generic"):
    expr_map = {}
    GATE_MAP = {
        "AND": And,
        "OR": Or,
        "XOR": Xor,
        "NAND": lambda a, b: Not(And(a, b)),
        "NOR": lambda a, b: Not(Or(a, b)),
        "XNOR": lambda a, b: Not(Xor(a, b)),
    }

    def get_symbol(name):
        if name.startswith("n") and len(name) > 1:
            return Not(symbols(name[1:]))
        return symbols(name)

    def build_expr(gate):
        in1, in2 = gate["inputs"]
        return GATE_MAP[gate["gate"]](
            expr_map.get(in1, get_symbol(in1)),
            expr_map.get(in2, get_symbol(in2))
        )

    for g in network:
        expr_map[g["name"]] = build_expr(g)

    if component_type == "full_adder":
        sum_expr = expr_map[network[-2]["name"]]
        carry_expr = expr_map[network[-1]["name"]]
        sum_simplified = simplify_with_patterns(sum_expr, [Xor(A, B, C)])
        carry_simplified = simplify_with_patterns(carry_expr, [
            Or(And(A, B), And(B, C), And(A, C))
        ])
        return {"SUM": sum_simplified, "CARRY": carry_simplified}

    elif component_type in ["d_latch", "d_flip_flop"]:
        q_expr = expr_map[network[-2]["name"]]
        nq_expr = expr_map[network[-1]["name"]]
        return {
            "Q": simplify_logic(q_expr, form="dnf"),
            "nQ": simplify_logic(nq_expr, form="dnf")
        }

    else:
        out_expr = expr_map[network[-1]["name"]]
        return {"OUT": simplify_logic(out_expr, form="dnf")}
