# grad_34_simp6.py
# Truth-table exact simplification with macro support (Phase5a-Turbo compatible)

from sympy import symbols, And, Or, Not, Xor, simplify_logic
from sympy.logic.boolalg import SOPform

_sym_cache = {}

def sym(name: str):
    s = _sym_cache.get(name)
    if s is None:
        s = symbols(name)
        _sym_cache[name] = s
    return s

def n(name: str):
    if name.startswith("n") and len(name) > 1:
        return Not(sym(name[1:]))
    return sym(name)

# ---------------- Gate semantics ----------------
def sym_AND(args):  return And(*args)
def sym_OR(args):   return Or(*args)
def sym_NAND(args): return Not(And(*args))
def sym_NOR(args):  return Not(Or(*args))
def sym_XOR(a,b):   return Xor(a,b)
def sym_XNOR(a,b):  return Not(Xor(a,b))

# --- Macro gates ---
def sym_HALF_SUM(a,b):     return Xor(a,b)
def sym_HALF_CARRY(a,b):   return And(a,b)
def sym_FULL_SUM(a,b,c):   return Xor(Xor(a,b),c)
def sym_FULL_CARRY(a,b,c): return Or(And(a,b), And(a,c), And(b,c))
def sym_MUX2(a,b,s):       return Or(And(Not(s), a), And(s, b))
def sym_EQ1(a,b):          return Not(Xor(a,b))
def sym_GT1(a,b):          return And(a, Not(b))
def sym_LT1(a,b):          return And(Not(a), b)

# ---------------- Mapping (Phase5a Turbo compatible) ----------------
GATE_SYM = {
    # primitives
    "AND":   ("var", sym_AND),
    "OR":    ("var", sym_OR),
    "NAND":  ("var", sym_NAND),
    "NOR":   ("var", sym_NOR),
    "XOR":   (2, sym_XOR), "XOR2": (2, sym_XOR),
    "XNOR":  (2, sym_XNOR), "XNOR2": (2, sym_XNOR),

    # macros
    "HALF_SUM":   (2, sym_HALF_SUM),
    "HALF_CARRY": (2, sym_HALF_CARRY),
    "FULL_SUM":   (3, sym_FULL_SUM),
    "FULL_CARRY": (3, sym_FULL_CARRY),
    "MUX2":       (3, sym_MUX2),  # (a,b,s)
    "EQ1":        (2, sym_EQ1),
    "GT1":        (2, sym_GT1),
    "LT1":        (2, sym_LT1),
}

# ---------------- Build symbolic expression map ----------------
def build_sym_expr_map(network):
    expr = {}
    for gate in network:
        gname = gate["gate"]
        if gname not in GATE_SYM:
            raise KeyError(f"Unsupported gate: {gname}")
        pins  = gate["inputs"]
        kind, fn = GATE_SYM[gname]

        args = [expr.get(tok, n(tok)) for tok in pins]

        if kind == "var":
            if len(args) < 2: args = args * 2
            if len(args) > 3: args = args[:3]
            out = fn(args)
        else:
            need = kind
            if len(args) < need: args = args + [args[-1]] * (need - len(args))
            if len(args) > need: args = args[:need]
            out = fn(*args)
        expr[gate["name"]] = out
    return expr

# ---------------- Truth-table exact simplification ----------------
def simplify_single_output(network, output_gate_name, input_names=None):
    emap = build_sym_expr_map(network)
    f_expr = emap[output_gate_name]

    bases = sorted({str(s) for s in f_expr.free_symbols if not str(s).startswith("n")})
    if input_names is None:
        input_names = bases
    else:
        input_names = [v for v in input_names if v in bases] or bases

    vars_syms = [sym(v) for v in input_names]
    n = len(vars_syms)
    
    minterms = []
    for idx in range(1 << n):
        assign = {vars_syms[i]: (idx >> (n - 1 - i)) & 1 for i in range(n)}
        val = int(bool(f_expr.xreplace(assign)))
        if val == 1:
            minterms.append(idx)

    if not minterms:
        return simplify_logic(False)

    sop = SOPform(vars_syms, minterms)
    sop2 = simplify_logic(sop, form="dnf", force=True)
    return sop2
