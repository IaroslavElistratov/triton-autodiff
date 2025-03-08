# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# # Make the autodiff headers visible to the rest of the codebase as if they were in the main include dir
# include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_SOURCE_DIR})

# # Add symbolic link for compatibility with old include path structure
# execute_process(
#   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include/autodiff/include/Dialect/MyArith/IR
# )

# Add subdirectories
add_subdirectory(include)
add_subdirectory(lib)

# if(TRITON_BUILD_PYTHON_MODULE)
#   add_triton_plugin(TritonAutoDiff ${CMAKE_CURRENT_SOURCE_DIR}/triton_autodiff.cc LINK_LIBS TritonToMyArith)
#   target_link_libraries(TritonAutoDiff PRIVATE Python3::Module pybind11::headers)
# endif()

# if(TRITON_BUILD_UT)
#   add_subdirectory(test)
# endif()
