def generate_n_bit_adder_vhdl(n, filename="n_bit_adder.vhd"):
    lines = []

    # === HEADER ===
    lines += [
        "library IEEE;",
        "use IEEE.STD_LOGIC_1164.ALL;",
        "",
        "entity n_bit_adder is",
        f"  Port (",
        f"    A     : in  STD_LOGIC_VECTOR({n-1} downto 0);",
        f"    B     : in  STD_LOGIC_VECTOR({n-1} downto 0);",
        f"    Sum   : out STD_LOGIC_VECTOR({n-1} downto 0);",
        f"    Carry : out STD_LOGIC",
        "  );",
        "end n_bit_adder;",
        ""
    ]

    # === ARCHITECTURE START ===
    lines += [
        "architecture Structural of n_bit_adder is",
        "",
        "  component full_adder_custom",
        "    Port (",
        "      A     : in  STD_LOGIC;",
        "      B     : in  STD_LOGIC;",
        "      C     : in  STD_LOGIC;",
        "      Sum   : out STD_LOGIC;",
        "      Carry : out STD_LOGIC",
        "    );",
        "  end component;",
        ""
    ]

    # Signal for internal carry wires
    if n > 1:
        lines.append(f"  signal carry : STD_LOGIC_VECTOR({n-2} downto 0);")
    lines.append("")

    # === BEGIN SECTION ===
    lines.append("begin")

    # First full adder (C = '0')
    lines += [
        f"  FA0: full_adder_custom port map(",
        f"    A => A(0),",
        f"    B => B(0),",
        f"    C => '0',",
        f"    Sum => Sum(0),",
        f"    Carry => carry(0)" if n > 1 else f"    Carry => Carry",
        "  );"
    ]

    # Remaining full adders (C comes from previous carry)
    for i in range(1, n):
        cin = f"carry({i-1})"
        cout = f"carry({i})" if i < n - 1 else "Carry"
        lines += [
            f"  FA{i}: full_adder_custom port map(",
            f"    A => A({i}),",
            f"    B => B({i}),",
            f"    C => {cin},",
            f"    Sum => Sum({i}),",
            f"    Carry => {cout}",
            "  );"
        ]

    # === END ===
    lines.append("end Structural;")

    with open(filename, "w") as f:
        f.write("\n".join(lines))

    print(f"\n✅ VHDL file generated: {filename}")


# --- RUN ---
if __name__ == "__main__":
    try:
        n = int(input("Enter the number of bits for the adder (n ≥ 1): "))
        if n < 1:
            raise ValueError
        generate_n_bit_adder_vhdl(n)
    except ValueError:
        print("❌ Please enter a valid positive integer (n ≥ 1).")
