library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity dlatch_custom is
  Port (
    D   : in  STD_LOGIC;
    En  : in  STD_LOGIC;
    Q   : out STD_LOGIC;
    Qn  : out STD_LOGIC
  );
end dlatch_custom;

architecture Structural of dlatch_custom is
  signal nD, nEN : STD_LOGIC;
  signal g0, g1, g2, g3, g4, g5 : STD_LOGIC;
begin
  nD <= not D;
  nEN <= not En;
  g0 <= nD xor D;
  g1 <= not (nD or nEN);
  g2 <= not (nD or nEN);
  g3 <= Q and nEN;
  g4 <= g3 xor g1;
  g5 <= not (g4 or g2);
  Q <= g4;
  Qn <= g5;
end Structural;