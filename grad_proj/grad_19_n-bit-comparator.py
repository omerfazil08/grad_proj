# comparator_nbit_main.py
from grad_18_comparator import comparator_1bit_logic
from typing import List


def to_bitvector(num: int, n: int) -> List[int]:
    return [int(b) for b in bin(num)[2:].zfill(n)]

def comparator_nbit_msb_first(a_bits: List[int], b_bits: List[int]) -> int:
    assert len(a_bits) == len(b_bits), "Bit lengths must match"
    for a, b in zip(a_bits, b_bits):
        if a != b:
            return comparator_1bit_logic(a, b)
    return 0  # Equal

def main():
    print("🧮 N-bit Binary Comparator (MSB → LSB)\n")

    n = int(input("Enter bit length (n): "))
    a_dec = int(input(f"Enter integer A (0 to {2**n - 1}): "))
    b_dec = int(input(f"Enter integer B (0 to {2**n - 1}): "))

    a_bits = to_bitvector(a_dec, n)
    b_bits = to_bitvector(b_dec, n)

    print(f"\nA = {a_dec} → bits: {a_bits}")
    print(f"B = {b_dec} → bits: {b_bits}")

    result = comparator_nbit_msb_first(a_bits, b_bits)

    print(f"\n✅ A > B? → {bool(result)}")

if __name__ == "__main__":
    main()
