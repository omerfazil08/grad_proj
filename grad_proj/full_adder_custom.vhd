library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity full_adder_custom is
  Port (
    A     : in  STD_LOGIC;
    B     : in  STD_LOGIC;
    C     : in  STD_LOGIC;
    Sum     : out STD_LOGIC;
    Carry     : out STD_LOGIC
  );
end full_adder_custom;

architecture Structural of full_adder_custom is
  signal nA, nB, nC : STD_LOGIC;
  signal g0, g1, g2, g3, g4, g5, g6, g7 : STD_LOGIC;
begin
  nA <= not A;
  nB <= not B;
  nC <= not C;
  g0 <= not (A and B);
  g1 <= A xor nC;
  g2 <= nB and nA;
  g3 <= not (g1 or g0);
  g4 <= not (g3 or C);
  g5 <= not (g0 and g2);
  g6 <= g1 xor nB;
  g7 <= not (g2 or g4);
  Sum <= g6;
  Carry <= g7;
end Structural;