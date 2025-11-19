# d_flip_flop_main.py

from grad_25_dflipflop import d_flip_flop_logic

def main():
    print("🔁 D Flip-Flop Simulation (Master-Slave via D Latches)\n")
    
    prev_master_q = 0
    prev_slave_q = 0

    while True:
        try:
            D = int(input("Enter D (0 or 1): "))
            CLK = int(input("Enter CLK (0 or 1): "))
            if D not in [0, 1] or CLK not in [0, 1]:
                raise ValueError

            master_q, slave_q = d_flip_flop_logic(D, CLK, prev_master_q, prev_slave_q)
            print(f"🧠 Master Q: {master_q} | 🧾 Flip-Flop Output (Slave Q): {slave_q}\n")

            prev_master_q = master_q
            prev_slave_q = slave_q

        except ValueError:
            print("Invalid input. Please enter only 0 or 1.\n")

if __name__ == "__main__":
    main()
