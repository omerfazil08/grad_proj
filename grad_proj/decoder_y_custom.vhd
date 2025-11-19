library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity decoder_y_custom is
  Port ( A, B, C : in STD_LOGIC;
         Y       : out STD_LOGIC );
end decoder_y_custom;

architecture Structural of decoder_y_custom is
  signal gate_out : STD_LOGIC;
begin
  gate_out <= not (A or C or B);
  Y <= gate_out;
end Structural;