add_triton_library(TritonToAutodiff
    TritonToAutodiffPass.cpp
    Utils.cpp
    UtilsIO.cpp

    DEPENDS
    TritonToAutodiffPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    AutodiffIR
)
