library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity half_adder_custom is
  Port (
    A     : in  STD_LOGIC;
    B     : in  STD_LOGIC;
    S     : out STD_LOGIC;
    C     : out STD_LOGIC
  );
end half_adder_custom;

architecture Structural of half_adder_custom is
  signal nA, nB : STD_LOGIC;
  signal g0, g1, g2, g3, g4, g5, g6, g7 : STD_LOGIC;
begin
  nA <= not A;
  nB <= not B;
  g0 <= nA or nB;
  g1 <= A or nB;
  g2 <= not (B or g0);
  g3 <= A xor g1;
  g4 <= g3 xor g2;
  g5 <= g0 or g3;
  g6 <= g5 xor g3;
  g7 <= B and g1;
  S <= g6;
  C <= g7;
end Structural;