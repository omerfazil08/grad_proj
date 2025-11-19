module circuit(input A, B, C, D, output Y0);
  // NOR
  assign g0 = ~(C | nD);
  // NOR
  assign g1 = ~(nA | nD);
  // NAND
  assign g2 = ~(nC & B);
  // NOR
  assign g3 = ~(nC | A);
  // OR
  assign g4 = g3 | g2;
  // NOR
  assign g5 = ~(B | A);
  // OR
  assign g6 = g2 | nD;
  // OR
  assign g7 = g4 | nC;
  // OR
  assign g8 = nA | g3;
  // XNOR
  assign g9 = ~(nA ^ g2);
  // NOR
  assign g10 = ~(g1 | nA);
  // NAND
  assign g11 = ~(g7 & D);
  // OR
  assign g12 = g1 | g4;
  // XOR
  assign g13 = g6 ^ g3;
endmodule