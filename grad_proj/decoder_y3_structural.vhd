library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity decoder_y3 is
    Port ( A, B, C : in STD_LOGIC;
           Y3 : out STD_LOGIC );
end decoder_y3;

architecture Structural of decoder_y3 is
    signal nA : STD_LOGIC;
    signal gate_out : STD_LOGIC;
begin
    nA <= not A;
    gate_out <= C and B and nA;
    Y3 <= gate_out;
end Structural;