# --- Logic Gates ---
def AND(a, b): return a & b
def OR(a, b): return a | b
def NOT(a): return 1 - a
def NOR(a, b): return NOT(OR(a, b))

# --- D-Latch Logic using SR-latch (based on NOR gates) ---
def d_latch(D, EN, prev_Q=0):
    """
    Simulates a basic D-Latch using an SR latch.
    prev_Q is the previously stored value (initially 0).
    """
    # Generate S and R inputs from D and EN
    Dn = NOT(D)
    S = AND(D, EN)
    R = AND(Dn, EN)

    # SR-latch using NOR gates
    Qn = NOR(S, prev_Q)
    Q = NOR(R, Qn)
    return Q, Qn

# --- Test D-Latch Behavior ---
def simulate_d_latch():
    print("D EN | Q Qn")
    prev_Q = 0
    for D in [0, 1]:
        for EN in [0, 1]:
            Q, Qn = d_latch(D, EN, prev_Q)
            print(f"{D}  {EN}  | {Q}  {Qn}")
            if EN == 1:
                prev_Q = Q  # Store the value when EN is high

simulate_d_latch()
