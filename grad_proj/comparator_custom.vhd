library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity comparator_custom is
  Port (
    A   : in  STD_LOGIC;
    B   : in  STD_LOGIC;
    OUT : out STD_LOGIC
  );
end comparator_custom;

architecture Structural of comparator_custom is
  signal nA, nB : STD_LOGIC;
  signal g0, g1, g2, g3 : STD_LOGIC;
begin
  nA <= not A;
  nB <= not B;
  g0 <= not (B xor A);
  g1 <= not (nB xor A);
  g2 <= not (g1 and A);
  g3 <= not (g0 or nA);
  OUT <= g3;
end Structural;