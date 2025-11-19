# grad_27_dflipflop_gates.py

from grad_28_dlatch_txt import simulate_d_latch_from_gates

import json

# Load gate config from file
with open("best_d_latch_gates.json") as f:
    evolved_gate_config = json.load(f)

# You can now reuse this in simulate_and_print or elsewhere

with open("best_d_latch_gates.json") as f:
    evolved_d_latch = json.load(f)

def d_flip_flop_logic(D: int, CLK: int, prev_master_q: int, prev_slave_q: int):
    master_enable = 1 - CLK
    master_q, _, master_signals = simulate_d_latch_from_gates(evolved_d_latch, D, master_enable, prev_master_q)

    slave_enable = CLK
    slave_q, _, slave_signals = simulate_d_latch_from_gates(evolved_d_latch, master_q, slave_enable, prev_slave_q)

    gate_outputs = {}
    for name, val in master_signals.items():
        gate_outputs[f"m_{name}"] = val
    for name, val in slave_signals.items():
        gate_outputs[f"s_{name}"] = val

    gate_outputs["qm"] = master_q
    gate_outputs["q"] = slave_q

    return master_q, slave_q, gate_outputs

def simulate_and_print():
    print("\n🧪 Gate-Level Simulation of D Flip-Flop (GA evolved)")
    print(" D CLK PrevM PrevS | Qm Q  | m_g4 (Qm) s_g4 (Q) | Full Gate States")
    print("-" * 70)

    for D in [0, 1]:
        for CLK in [0, 1]:
            for prev_master_q in [0, 1]:
                for prev_slave_q in [0, 1]:
                    qm, q, gates = d_flip_flop_logic(D, CLK, prev_master_q, prev_slave_q)

                    print(f" {D}   {CLK}    {prev_master_q}     {prev_slave_q}   | {qm}  {q}  |   {gates['m_g4']}        {gates['s_g4']}     | ", end="")
                    print(", ".join(f"{k}={v}" for k, v in gates.items() if k not in ['qm', 'q']))

with open("d_flip_flop_output.txt", "w") as f:
    for D in [0, 1]:
        for CLK in [0, 1]:
            for prev_master_q in [0, 1]:
                for prev_slave_q in [0, 1]:
                    qm, q, gates = d_flip_flop_logic(D, CLK, prev_master_q, prev_slave_q)
                    line = f"{D} {CLK} {prev_master_q} {prev_slave_q} | {qm} {q} | m_g4={gates.get('m_g4')} s_g4={gates.get('s_g4')}\n"
                    print(line.strip())
                    f.write(line)

if __name__ == "__main__":
    simulate_and_print()
