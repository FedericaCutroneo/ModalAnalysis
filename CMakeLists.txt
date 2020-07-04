cmake_minimum_required (VERSION 3.0)
project(ModalAnalysis)
 
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Spectra REQUIRED)
 
add_executable(TestLOBPCG TestLOBPCG.cpp)

target_link_libraries(TestLOBPCG Eigen3::Eigen)
target_link_libraries(TestLOBPCG Spectra::Spectra)
target_compile_options(TestLOBPCG PUBLIC "/bigobj")

add_executable(SymGEigsCholesky SymGEigsCholesky.cpp)

target_link_libraries(SymGEigsCholesky Eigen3::Eigen)
target_link_libraries(SymGEigsCholesky Spectra::Spectra)

