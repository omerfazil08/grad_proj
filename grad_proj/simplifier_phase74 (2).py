# simplifier_phase74.py
# Updated to return SymPy objects directly (Avoiding String Parser Crashes)

from sympy import symbols, And, Or, Not, Xor, simplify, Implies, Equivalent
from typing import List, Dict

print("✅ Loaded simplifier_phase74 (Direct Object Mode)")

# --- Symbol cache ---
_sym_cache = {}

def sym(name: str):
    if name not in _sym_cache:
        _sym_cache[name] = symbols(name)
    return _sym_cache[name]

# --- Gate Factory ---
GATE_FACTORY = {
    "AND": lambda args: And(*args),
    "OR": lambda args: Or(*args),
    "NOT": lambda args: Not(args[0]),
    "XOR": lambda args: Xor(*args),
    "XOR2": lambda args: Xor(*args),
    "XNOR": lambda args: Not(Xor(*args)),
    "XNOR2": lambda args: Not(Xor(*args)),
    "NAND": lambda args: Not(And(*args)),
    "NOR": lambda args: Not(Or(*args)),
    "MUX2": lambda args: Or(And(Not(args[0]), args[1]), And(args[0], args[2]))
}

def build_sym_expr_map(network: List[Dict[str, object]]) -> Dict[str, object]:
    expr = {}
    input_symbols = set()
    for gate in network:
        for tok in gate.get("inputs", []):
            if isinstance(tok, str) and tok.startswith("A"):
                input_symbols.add(tok)
    for in_name in input_symbols:
        expr[in_name] = sym(in_name)

    unresolved = list(network)
    progress = True
    
    while unresolved and progress:
        progress = False
        remaining = []
        for gate in unresolved:
            out_name = gate.get("output", gate.get("name"))
            gtype = gate.get("type")
            raw_inputs = gate.get("inputs", [])
            
            args = []
            resolvable = True
            for tok in raw_inputs:
                if tok in expr:
                    args.append(expr[tok])
                elif isinstance(tok, str) and tok.startswith("A"):
                    expr[tok] = sym(tok)
                    args.append(expr[tok])
                else:
                    resolvable = False
                    break
            
            if not resolvable:
                remaining.append(gate)
                continue

            factory = GATE_FACTORY.get(gtype, None)
            try:
                if factory:
                    built = factory(args)
                else:
                    built = And(*args)
                expr[out_name] = built
                progress = True
            except Exception:
                expr[out_name] = sym(out_name)
                progress = True

        unresolved = remaining

    for gate in unresolved:
        out_name = gate.get("output", gate.get("name"))
        if out_name not in expr:
            expr[out_name] = sym(out_name)

    return expr

def simplify_genome(best_individual: List[Dict], input_names: List[str], output_gate_names: List[str]) -> Dict[str, object]:
    """
    Returns a dictionary of SymPy EXPRESSION OBJECTS (not strings).
    This avoids parser ambiguity with XOR (^ vs **).
    """
    expr_map = build_sym_expr_map(best_individual)
    results = {}

    for i, out_gate in enumerate(output_gate_names):
        key = f"Output {i+1} ({out_gate})"
        if out_gate not in expr_map:
            results[key] = sym("False")
            continue
        raw_expr = expr_map[out_gate]

        try:
            # Algebraic simplification (Preserves XOR)
            final_expr = simplify(raw_expr)
            results[key] = final_expr
        except Exception:
            results[key] = raw_expr

    return results