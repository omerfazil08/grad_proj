library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity d-latch_custom is
  Port (
    D     : in  STD_LOGIC;
    En    : in  STD_LOGIC;
    Q     : out STD_LOGIC
  );
end d-latch_custom;

architecture Structural of d-latch_custom is
  signal nD : STD_LOGIC;
  signal g0, g1, g2, g3, g4, g5 : STD_LOGIC;
begin
  nD <= not D;
  g0 <= nEN xor EN;
  g1 <= not (nD or D);
  g2 <= nD or EN;
  g3 <= not (nD and D);
  g4 <= not (nEN xor Q);
  g5 <= not (Q and g0);
  Q <= g5;
end Structural;