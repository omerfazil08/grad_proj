library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity evolved_circuit is
    Port (
        I0 : in  STD_LOGIC;
        I1 : in  STD_LOGIC;
        I2 : in  STD_LOGIC;
        I3 : in  STD_LOGIC;
        I4 : in  STD_LOGIC;
        I5 : in  STD_LOGIC;
        O0 : out STD_LOGIC;
        O1 : out STD_LOGIC;
        O2 : out STD_LOGIC;
        O3 : out STD_LOGIC
    );
end evolved_circuit;

architecture Structural of evolved_circuit is
    signal w_G0, w_G1, w_G2, w_G3, w_G4, w_G5, w_G6, w_G7, w_G8, w_G9, w_G10, w_G11, w_G12, w_G13, w_G14, w_G15, w_G16, w_G17, w_G18, w_G19, w_G20, w_G21, w_G22, w_G23, w_G24, w_G25, w_G26, w_G27, w_G28, w_G29, w_G30, w_G31, w_G32, w_G33, w_G34, w_G35, w_G36, w_G37, w_G38, w_G39 : STD_LOGIC;
begin
    w_G0 <= (not I0 and I5) or (I0 and I5);
    w_G1 <= I3 xor I0;
    w_G2 <= not (I5 xor I4);
    w_G3 <= I0 and I3 and I1;
    w_G4 <= not (I3 xor I2);
    w_G5 <= (I4 and w_G1) or (I4 and I1) or (w_G1 and I1);
    w_G6 <= I4 xor I3;
    w_G7 <= I2 and I0;
    w_G8 <= not (I5 xor I2);
    w_G9 <= not w_G1;
    w_G10 <= (not I5 and w_G0) or (I5 and I2);
    w_G11 <= w_G5 xor w_G1;
    w_G12 <= w_G8 and w_G11 and I1;
    w_G13 <= not (I1 xor w_G0);
    w_G14 <= w_G9 and I3 and w_G4;
    w_G15 <= w_G10 xor w_G6;
    w_G16 <= I5 xor w_G6 xor w_G13;
    w_G17 <= (not w_G0 and I1) or (w_G0 and w_G13);
    w_G18 <= w_G14 and w_G15;
    w_G19 <= w_G11 xor w_G2;
    w_G20 <= I4 xor w_G15 xor w_G17;
    w_G21 <= w_G19 xor w_G10 xor w_G3;
    w_G22 <= w_G3 xor w_G13;
    w_G23 <= (I2 and w_G8) or (I2 and w_G0) or (w_G8 and w_G0);
    w_G24 <= not (I1 xor w_G2);
    w_G25 <= w_G11 xor w_G21;
    w_G26 <= not (w_G19 and I0);
    w_G27 <= w_G0 xor w_G6;
    w_G28 <= (w_G17 and I4) or (w_G17 and w_G10) or (I4 and w_G10);
    w_G29 <= (w_G1 and w_G4) or (w_G1 and I0) or (w_G4 and I0);
    w_G30 <= w_G14 and w_G7;
    w_G31 <= w_G4 and w_G26;
    w_G32 <= w_G25 or I3 or w_G19;
    w_G33 <= not (w_G4 and I3 and w_G21);
    w_G34 <= not (w_G33 and w_G28 and w_G18);
    w_G35 <= w_G26 xor w_G27;
    w_G36 <= I2 xor I5;
    w_G37 <= w_G23 xor w_G13 xor w_G2;
    w_G38 <= w_G28 xor w_G1;
    w_G39 <= (I3 and I0) or (I3 and w_G28) or (I0 and w_G28);

    -- Outputs
    O0 <= w_G36;
    O1 <= w_G37;
    O2 <= w_G38;
    O3 <= w_G39;
end Structural;