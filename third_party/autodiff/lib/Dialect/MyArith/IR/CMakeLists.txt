add_triton_library(MyArithIR
  Dialect.cpp

  DEPENDS
  MyArithTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
)
