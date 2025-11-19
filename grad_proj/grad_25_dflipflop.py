# grad_18_d_flip_flop.py

from grad_24_dlatch import d_latch_logic

def d_flip_flop_logic(D: int, CLK: int, prev_master_q: int, prev_slave_q: int):
    """
    Simulate D Flip-Flop using two D latches (master-slave configuration).
    :param D: Input value
    :param CLK: Clock signal (0 or 1)
    :param prev_master_q: Previous state of master latch
    :param prev_slave_q: Previous state of slave latch
    :return: Tuple (master_q, slave_q)
    """

    # Master latch enabled when CLK = 0 (inverted)
    master_enable = 1 - CLK
    master_q, _ = d_latch_logic(D, master_enable, prev_master_q)

    # Slave latch enabled when CLK = 1
    slave_enable = CLK
    slave_q, _ = d_latch_logic(master_q, slave_enable, prev_slave_q)

    return master_q, slave_q  # slave_q is the output of the flip-flop (Q)
