library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity mux_custom is
  Port (
    A     : in  STD_LOGIC;
    B     : in  STD_LOGIC;
    SEL   : in  STD_LOGIC;
    OUT   : out STD_LOGIC
  );
end mux_custom;

architecture Structural of mux_custom is
  signal nA, nB, nSEL : STD_LOGIC;
  signal g0, g1, g2, g3, g4, g5, g6, g7 : STD_LOGIC;
begin
  nA <= not A;
  nB <= not B;
  nSEL <= not SEL;
  g0 <= B and SEL;
  g1 <= g0 xor SEL;
  g2 <= nA and g0;
  g3 <= B xor g1;
  g4 <= SEL and nB;
  g5 <= g3 xor nSEL;
  g6 <= not (g1 or nA);
  g7 <= g6 or g0;
  OUT <= g7;
end Structural;