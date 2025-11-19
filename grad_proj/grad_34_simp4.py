# grad_34_simp3.py
# Macro-aware Boolean simplifier for Phase 5.a
from sympy import symbols, simplify_logic, Not, And, Or, Xor

# ===== Sympy symbol helper =====
def get_symbol(name: str):
    if name.startswith("n") and len(name) == 2 and name[1].isalpha():
        return Not(symbols(name[1]))
    if len(name) == 1 and name.isalpha():
        return symbols(name)
    # gate outputs will be handled by expr_map
    return symbols(name)

# ===== Gate -> Sympy mapping =====
def sym_AND(args):  return And(*args)
def sym_OR(args):   return Or(*args)
def sym_XOR2(a, b): return Xor(a, b)
def sym_XNOR2(a, b): return Not(Xor(a, b))
def sym_NAND(args): return Not(And(*args))
def sym_NOR(args):  return Not(Or(*args))

# Macros
def sym_HALF_SUM(a, b):   return Xor(a, b)
def sym_HALF_CARRY(a, b): return And(a, b)
def sym_FULL_SUM(a, b, c):   return Xor(Xor(a, b), c)
def sym_FULL_CARRY(a, b, c): return Or(Or(And(a, b), And(a, c)), And(b, c))
def sym_MUX2(sel, a, b):  return Or(And(Not(sel), a), And(sel, b))
def sym_EQ1(a, b):        return Not(Xor(a, b))
def sym_GT1(a, b):        return And(a, Not(b))
def sym_LT1(a, b):        return And(Not(a), b)

# Registry describing arity & builder
GATE_BUILDERS = {
    "AND":   ("2or3", lambda ins: sym_AND(ins)),
    "OR":    ("2or3", lambda ins: sym_OR(ins)),
    "NAND":  ("2or3", lambda ins: sym_NAND(ins)),
    "NOR":   ("2or3", lambda ins: sym_NOR(ins)),
    "XOR":   (2,      lambda ins: sym_XOR2(ins[0], ins[1])),
    "XNOR":  (2,      lambda ins: sym_XNOR2(ins[0], ins[1])),

    "HALF_SUM":   (2, lambda ins: sym_HALF_SUM(ins[0], ins[1])),
    "HALF_CARRY": (2, lambda ins: sym_HALF_CARRY(ins[0], ins[1])),
    "FULL_SUM":   (3, lambda ins: sym_FULL_SUM(ins[0], ins[1], ins[2])),
    "FULL_CARRY": (3, lambda ins: sym_FULL_CARRY(ins[0], ins[1], ins[2])),
    "MUX2":       (3, lambda ins: sym_MUX2(ins[0], ins[1], ins[2])),
    "EQ1":        (2, lambda ins: sym_EQ1(ins[0], ins[1])),
    "GT1":        (2, lambda ins: sym_GT1(ins[0], ins[1])),
    "LT1":        (2, lambda ins: sym_LT1(ins[0], ins[1])),
}

def _coerce_args(name: str, arg_exprs):
    # Handle base gates that allow 2 or 3 inputs; trim or expand if needed
    arity = GATE_BUILDERS[name][0]
    if arity == "2or3":
        # Keep as-is (2 or 3 supported); if more, trim; if <2, duplicate last
        if len(arg_exprs) < 2:
            arg_exprs = list(arg_exprs) + [arg_exprs[-1]] * (2 - len(arg_exprs))
        elif len(arg_exprs) > 3:
            arg_exprs = list(arg_exprs[:3])
        return arg_exprs
    need = int(arity)
    if len(arg_exprs) != need:
        # trim or pad with last
        if len(arg_exprs) > need:
            return list(arg_exprs[:need])
        else:
            return list(arg_exprs) + [arg_exprs[-1]] * (need - len(arg_exprs))
    return arg_exprs

def build_expr(network):
    expr_map = {}
    # seed primary inputs if referenced
    for gate in network:
        for inp in gate["inputs"]:
            if inp not in expr_map and (len(inp) == 1 and inp.isalpha()):
                expr_map[inp] = get_symbol(inp)
            if inp not in expr_map and (len(inp) == 2 and inp[0] == "n" and inp[1].isalpha()):
                expr_map[inp] = get_symbol(inp)

    for gate in network:
        name = gate["gate"]
        ins = []
        for w in gate["inputs"]:
            ins.append(expr_map.get(w, get_symbol(w)))
        if name not in GATE_BUILDERS:
            # fallback treat as AND of inputs
            expr_map[gate["name"]] = And(*ins)
            continue
        arity, builder = GATE_BUILDERS[name]
        ins = _coerce_args(name, ins)
        expr_map[gate["name"]] = builder(ins)
    return expr_map

def simplify_single_output(network, output_gate_name):
    expr_map = build_expr(network)
    final_expr = expr_map[output_gate_name]
    # Use DNF for readability; SOP-like
    return simplify_logic(final_expr, form="dnf")
