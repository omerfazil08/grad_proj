"""
grad_34_simp7.py
Enhanced Boolean simplifier and recognizer for evolved logic networks.
- Supports all primitive and macro gates
- Detects XOR / XNOR / parity patterns
- Outputs both symbolic and Verilog-friendly forms
"""

from sympy import symbols, simplify_logic, Not, And, Or, Xor, Equivalent, simplify
from sympy.logic.boolalg import SOPform
import itertools


# ---------------------------------------------------------------------
# 🎛 Supported Gates (primitive + macro)
# ---------------------------------------------------------------------
GATE_SYM = {
    # 2-input basic gates
    "AND": lambda *a: And(*a),
    "OR": lambda *a: Or(*a),
    "XOR": lambda *a: Xor(*a),
    "XNOR": lambda *a: Not(Xor(*a)),
    "NAND": lambda *a: Not(And(*a)),
    "NOR": lambda *a: Not(Or(*a)),

    # Variable-arity safe versions
    "AND2": lambda a, b: And(a, b),
    "OR2": lambda a, b: Or(a, b),
    "XOR2": lambda a, b: Xor(a, b),
    "XNOR2": lambda a, b: Not(Xor(a, b)),

    # Common macro structures
    "HALF_SUM": lambda a, b, *rest: Xor(a, b),
    "HALF_CARRY": lambda a, b, *rest: And(a, b),
    "FULL_SUM": lambda a, b, c, *rest: Xor(a, b, c),
    "FULL_CARRY": lambda a, b, c, *rest: Or(And(a, b), And(a, c), And(b, c)),

    # Comparators
    "EQ1": lambda a, b, *rest: Not(Xor(a, b)),
    "GT1": lambda a, b, *rest: And(a, Not(b)),
    "LT1": lambda a, b, *rest: And(Not(a), b),

    # MUX2: 3-input, (sel=c)
    "MUX2": lambda a, b, s, *rest: Or(And(Not(s), a), And(s, b)),
}


# ---------------------------------------------------------------------
# 🧩 Utilities
# ---------------------------------------------------------------------
def get_symbol(name):
    """Convert GA variable name to sympy symbol or negated form."""
    if name.startswith("n") and len(name) > 1:
        return Not(symbols(name[1:]))
    return symbols(name)


def build_sym_expr_map(network):
    """
    Build a sympy expression map from evolved network structure.
    Each gate entry is assumed to have 'gate', 'inputs', and 'name'.
    """
    emap = {}
    for gate in network:
        gname = gate["gate"]
        fn = GATE_SYM.get(gname)
        if fn is None:
            # fallback to direct AND for unknown types
            fn = lambda *a: And(*a)
        in_syms = []
        for inp in gate["inputs"]:
            if inp in emap:
                in_syms.append(emap[inp])
            else:
                in_syms.append(get_symbol(inp))
        try:
            emap[gate["name"]] = fn(*in_syms)
        except TypeError:
            # handle mismatch in arity gracefully
            emap[gate["name"]] = fn(*in_syms[:3])
    return emap


# ---------------------------------------------------------------------
# 🧮 Simplification Logic
# ---------------------------------------------------------------------
def recognize_special_forms(expr, inputs):
    """
    Detect XOR, XNOR, parity, or majority patterns.
    If found, return simplified symbolic form.
    """
    # Evaluate truth table
    var_syms = [symbols(i) for i in inputs]
    tt = []
    for vals in itertools.product([0, 1], repeat=len(var_syms)):
        mapping = dict(zip(var_syms, vals))
        out = int(bool(expr.xreplace(mapping)))
        tt.append(out)

    # Check for XOR parity pattern
    xor_tt = [int(sum(vals) % 2) for vals in itertools.product([0, 1], repeat=len(var_syms))]
    if tt == xor_tt:
        return Xor(*var_syms)

    # XNOR (even parity)
    xnor_tt = [int((sum(vals) + 1) % 2) for vals in itertools.product([0, 1], repeat=len(var_syms))]
    if tt == xnor_tt:
        return Not(Xor(*var_syms))

    # Majority
    maj_tt = [int(sum(vals) >= (len(var_syms) // 2 + 1)) for vals in itertools.product([0, 1], repeat=len(var_syms))]
    if tt == maj_tt:
        return Or(*[And(*combo) for combo in itertools.combinations(var_syms, 2)])

    # No special pattern
    return simplify_logic(expr, form="dnf", force=True)


def simplify_single_output(network, output_gate_name, input_names=None):
    """
    Simplify a single output expression from the evolved network.
    - Returns a sympy expression that matches the truth table exactly
    - Recognizes XOR/XNOR/majority patterns
    """
    emap = build_sym_expr_map(network)
    final_expr = emap[output_gate_name]
    if input_names is None:
        input_names = ["A", "B", "C", "D"]
    try:
        simplified = recognize_special_forms(final_expr, input_names)
    except Exception:
        simplified = simplify_logic(final_expr, form="dnf", force=True)
    return simplified


# ---------------------------------------------------------------------
# 🧱 Verilog Export Helpers
# ---------------------------------------------------------------------
def expr_to_verilog(expr):
    """Convert sympy boolean expression to Verilog syntax."""
    s = str(expr)
    s = s.replace("~", "!")
    s = s.replace("&", "&&")
    s = s.replace("|", "||")
    s = s.replace("Xor", "^")
    s = s.replace("Not", "!")
    s = s.replace("Or", "||")
    s = s.replace("And", "&&")
    return s


def export_to_verilog(module_name, inputs, outputs, simplified_exprs):
    """
    Generate a full Verilog module string.
    - inputs: list of input names
    - outputs: list of output names
    - simplified_exprs: list of sympy expressions for each output
    """
    lines = []
    lines.append(f"module {module_name}(")
    lines.append(f"    input {', '.join(inputs)},")
    lines.append(f"    output {', '.join(outputs)}")
    lines.append(");")

    for out_name, expr in zip(outputs, simplified_exprs):
        verilog_expr = expr_to_verilog(expr)
        lines.append(f"    assign {out_name} = {verilog_expr};")

    lines.append("endmodule")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# ✅ Example Test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example: 3-input full adder (Sum + Carry)
    test_net = [
        {"gate": "XOR", "inputs": ["A", "B"], "name": "g0"},
        {"gate": "XOR", "inputs": ["g0", "C"], "name": "Sum"},
        {"gate": "FULL_CARRY", "inputs": ["A", "B", "C"], "name": "Carry"},
    ]

    sum_expr = simplify_single_output(test_net, "Sum", ["A", "B", "C"])
    carry_expr = simplify_single_output(test_net, "Carry", ["A", "B", "C"])

    print("Sum simplified:", sum_expr)
    print("Carry simplified:", carry_expr)
    print("\n--- Generated Verilog ---\n")
    print(export_to_verilog("full_adder", ["A", "B", "C"], ["Sum", "Carry"], [sum_expr, carry_expr]))
