library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity n_bit_adder is
  Port (
    A     : in  STD_LOGIC_VECTOR(2 downto 0);
    B     : in  STD_LOGIC_VECTOR(2 downto 0);
    Sum   : out STD_LOGIC_VECTOR(2 downto 0);
    Carry : out STD_LOGIC
  );
end n_bit_adder;

architecture Structural of n_bit_adder is

  component full_adder_custom
    Port (
      A     : in  STD_LOGIC;
      B     : in  STD_LOGIC;
      C     : in  STD_LOGIC;
      Sum   : out STD_LOGIC;
      Carry : out STD_LOGIC
    );
  end component;

  signal carry : STD_LOGIC_VECTOR(1 downto 0);

begin
  FA0: full_adder_custom port map(
    A => A(0),
    B => B(0),
    C => '0',
    Sum => Sum(0),
    Carry => carry(0)
  );
  FA1: full_adder_custom port map(
    A => A(1),
    B => B(1),
    C => carry(0),
    Sum => Sum(1),
    Carry => carry(1)
  );
  FA2: full_adder_custom port map(
    A => A(2),
    B => B(2),
    C => carry(1),
    Sum => Sum(2),
    Carry => Carry
  );
end Structural;