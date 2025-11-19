# grad_34_simp5.py
# Truth-table exact simplification with macro support and fixed port orders.

from sympy import symbols, And, Or, Not, Xor, simplify_logic
from sympy.logic.boolalg import SOPform

# ---------- Symbol cache ----------
_sym_cache = {}

def sym(name: str):
    s = _sym_cache.get(name)
    if s is None:
        s = symbols(name)
        _sym_cache[name] = s
    return s

def n(name: str):
    # nA -> ~A  (already represented as "nA" in genomes)
    if name.startswith("n") and len(name) > 1:
        return Not(sym(name[1:]))
    return sym(name)

# ---------- Gate to SymPy with fixed semantics ----------
def sym_AND(args):  return And(*args)
def sym_OR(args):   return Or(*args)
def sym_NAND(args): return Not(And(*args))
def sym_NOR(args):  return Not(Or(*args))
def sym_XOR(a,b):   return Xor(a,b)
def sym_XNOR(a,b):  return Not(Xor(a,b))

# Macros (fixed arity, fixed port order)
# HALF_SUM(a,b) = a ^ b
# HALF_CARRY(a,b) = a & b
# FULL_SUM(a,b,c) = a ^ b ^ c
# FULL_CARRY(a,b,c) = (a&b) | (a&c) | (b&c)
# MUX2(a,b,s) = (~s & a) | (s & b)
# EQ1(a,b) = XNOR(a,b)
# GT1(a,b) = a & ~b
# LT1(a,b) = ~a & b

def sym_HALF_SUM(a,b):     return Xor(a,b)
def sym_HALF_CARRY(a,b):   return And(a,b)
def sym_FULL_SUM(a,b,c):   return Xor(Xor(a,b),c)
def sym_FULL_CARRY(a,b,c): return Or(And(a,b), And(a,c), And(b,c))
def sym_MUX2(a,b,s):       return Or(And(Not(s), a), And(s, b))
def sym_EQ1(a,b):          return Not(Xor(a,b))
def sym_GT1(a,b):          return And(a, Not(b))
def sym_LT1(a,b):          return And(Not(a), b)

# ---------- Mapping ----------
# We accept variable arity for AND/OR/NAND/NOR (2 or 3) as in your GA;
# XOR/XNOR are strictly 2-input here.
GATE_SYM = {
    "AND":   ("var", sym_AND),
    "OR":    ("var", sym_OR),
    "NAND":  ("var", sym_NAND),
    "NOR":   ("var", sym_NOR),
    "XOR":   (2,     sym_XOR),
    "XNOR":  (2,     sym_XNOR),

    "HALF_SUM":   (2, sym_HALF_SUM),
    "HALF_CARRY": (2, sym_HALF_CARRY),
    "FULL_SUM":   (3, sym_FULL_SUM),
    "FULL_CARRY": (3, sym_FULL_CARRY),
    "MUX2":       (3, sym_MUX2),    # (a, b, s)  << IMPORTANT
    "EQ1":        (2, sym_EQ1),
    "GT1":        (2, sym_GT1),
    "LT1":        (2, sym_LT1),
}

# ---------- Build symbolic expression for a network ----------
def build_sym_expr_map(network):
    expr = {}
    for gate in network:
        gname = gate["gate"]
        pins  = gate["inputs"]  # strings
        kind, fn = GATE_SYM[gname]

        # Convert each input token to a SymPy expression
        args = []
        for tok in pins:
            args.append(expr.get(tok, n(tok)))

        if kind == "var":
            # clamp to 2..3 inputs safely
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

# ---------- Truth-table driven simplify ----------
def simplify_single_output(network, output_gate_name, input_names=None):
    """
    Returns a SymPy expression that matches the evolved network's output
    exactly on the given truth table domain, using SOPform.

    input_names: optional list of variable order (e.g., ["A","B","C","D"]).
                 If None, it is inferred from present symbols (sorted).
    """
    # 1) Build symbolic expr map
    emap = build_sym_expr_map(network)
    f_expr = emap[output_gate_name]

    # 2) Discover variable set (A..H and their nX counterparts)
    #    Only base symbols (A,B,C,...) are used in SOPform.
    bases = sorted({str(s) for s in f_expr.free_symbols if not str(s).startswith("n")})
    if input_names is None:
        input_names = bases
    else:
        # keep only used vars but preserve provided order
        input_names = [v for v in input_names if v in bases] or bases

    vars_syms = [sym(v) for v in input_names]

    # 3) Enumerate truth table by evaluating f_expr on all input combos
    #    Build list of minterm indices where the function is 1.
    minterms = []
    n = len(vars_syms)
    # Map order: same as main code (lexicographic in your UI)
    for idx in range(1 << n):
        # Assign values
        assign = {}
        for i, v in enumerate(vars_syms):
            assign[v] = (idx >> (n - 1 - i)) & 1

        val = int(bool(f_expr.xreplace(assign)))
        if val == 1:
            minterms.append(idx)

    if not minterms:
        # Function always 0
        return simplify_logic(False)

    # 4) SOP minimal form that is truth-table exact
    sop = SOPform(vars_syms, minterms)

    # Optional: re-run simplify_logic to compact small cases (preserves truth table)
    sop2 = simplify_logic(sop, form="dnf", force=True)
    return sop2
