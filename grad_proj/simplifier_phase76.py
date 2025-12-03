# simplifier_phase76.py
# Advanced Simplifier: BRUTE FORCE DNF.
# Uses sympy.logic.boolalg.to_dnf to guarantee no XOR symbols exist.

from sympy import symbols, And, Or, Not, Xor, simplify_logic
from sympy.logic.boolalg import to_dnf

print("✅ Loaded simplifier_phase76 (Brute Force DNF)")

# --- Symbol Management ---
_sym_cache = {}

def sym(name: str):
    if name not in _sym_cache:
        _sym_cache[name] = symbols(name)
    return _sym_cache[name]

def get_input_sym(tok: str):
    if tok.startswith("n") and len(tok) > 1 and tok[1].isupper():
        return Not(sym(tok[1:]))
    return sym(tok)

# --- Gate Logic Mappers ---
def sym_AND(args):  return And(*args)
def sym_OR(args):   return Or(*args)
def sym_NOT(a):     return Not(a)
def sym_XOR(a, b):  return Xor(a, b)
def sym_XNOR(a, b): return Not(Xor(a, b))
def sym_NAND(args): return Not(And(*args))
def sym_NOR(args):  return Not(Or(*args))
def sym_MUX2(s, a, b): return Or(And(Not(s), a), And(s, b))
def sym_HALF_SUM(a, b): return Xor(a, b)
def sym_HALF_CARRY(a, b): return And(a, b)
def sym_FULL_SUM(a, b, c): return Xor(Xor(a, b), c)
def sym_FULL_CARRY(a, b, c): return Or(And(a, b), And(a, c), And(b, c))
def sym_EQ1(a, b): return Not(Xor(a, b))

GATE_SYM_MAP = {
    "AND": sym_AND, "OR": sym_OR, "NOT": lambda a: sym_NOT(a[0]),
    "XOR": lambda a: sym_XOR(a[0], a[1]), "XOR2": lambda a: sym_XOR(a[0], a[1]),
    "XNOR": lambda a: sym_XNOR(a[0], a[1]), "XNOR2": lambda a: sym_XNOR(a[0], a[1]),
    "NAND": sym_NAND, "NOR": sym_NOR, "EQ1": lambda a: sym_EQ1(a[0], a[1]),
    "MUX2": lambda a: sym_MUX2(a[0], a[1], a[2]),
    "HALF_SUM": lambda a: sym_HALF_SUM(a[0], a[1]),
    "HALF_CARRY": lambda a: sym_HALF_CARRY(a[0], a[1]),
    "FULL_SUM": lambda a: sym_FULL_SUM(a[0], a[1], a[2]),
    "FULL_CARRY": lambda a: sym_FULL_CARRY(a[0], a[1], a[2]),
}

def build_sym_expr_map(network):
    expr = {}
    for gate in network:
        gname = gate["type"]
        inputs = gate["inputs"]
        out_name = gate["output"]
        
        arg_exprs = []
        for inp in inputs:
            if inp in expr:
                arg_exprs.append(expr[inp])
            else:
                arg_exprs.append(get_input_sym(inp))
        
        if gname in GATE_SYM_MAP:
            try:
                sym_out = GATE_SYM_MAP[gname](arg_exprs)
                expr[out_name] = sym_out
            except IndexError:
                expr[out_name] = get_input_sym("ERR")
        else:
            expr[out_name] = And(*arg_exprs)
    return expr

def simplify_genome(best_individual, input_names, output_gate_names):
    expr_map = build_sym_expr_map(best_individual)
    results = {}
    
    for i, out_gate in enumerate(output_gate_names):
        if out_gate not in expr_map: continue
        raw_expr = expr_map[out_gate]
        
        try:
            final_expr = to_dnf(raw_expr, simplify=True)
            results[f"Output {i+1} ({out_gate})"] = str(final_expr)
        except Exception:
            results[f"Output {i+1} ({out_gate})"] = str(raw_expr)
        
    return results