# Add subdirectories
add_subdirectory(include)
add_subdirectory(lib)

# Python bindings
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonAutoDiff ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_autodiff.cc LINK_LIBS TritonToMyArith)
  target_link_libraries(TritonAutoDiff PRIVATE Python3::Module pybind11::headers)
endif()
