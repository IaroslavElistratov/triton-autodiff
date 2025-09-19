add_triton_library(AutodiffIR
  Dialect.cpp

  DEPENDS
  AutodiffTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
)
