add_triton_library(TritonToMyArith
    TritonToMyArithPass.cpp

    DEPENDS
    TritonToMyArithPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    MyArithIR
)
