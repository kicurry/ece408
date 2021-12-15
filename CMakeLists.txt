############################################################################################
# <summary> CMakeLists.txt for ECE408 								</summary>
# <date>    2021-12-10          									</date>
############################################################################################
project(ECE408)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(TEMPLATE "templates")
set(REFERENCE "ref")
set(IMPLEMENT "imp")
find_package(CUDA REQUIRED)
# collect source files

if(CUDA_FOUND)
    # compared to class settings, we let NVidia's FindCUDA CMake detect 
    # whether to build x64.  We tell it to support most devices, though, 
    # to make sure more people can easily run class code without knowing 
    # about this compiler argument
    if (MSVC)
        set(CUDA_NVCC_FLAGS "")
    else (MSVC)
        set(CUDA_NVCC_FLAGS "-gencode;arch=compute_86,code=sm_86;")
    endif (MSVC)

    # add -Wextra compiler flag for gcc compilations
    if (UNIX)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wall;")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
    endif (UNIX)

    # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
	include_directories(${CMAKE_SOURCE_DIR})
	include_directories(${CUDA_INCLUDE_DIRS})

else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()

file( GLOB  wbhdr lib/*.hpp lib/*.h )
file( GLOB  wbsrc lib/*.cpp lib/*.c )

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories("lib/")

# Add Template code 
# CUDA_ADD_EXECUTABLE(MP0_Template ${TEMPLATE}/mp0_template.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP1_Template ${TEMPLATE}/mp1_template.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP2_Template ${TEMPLATE}/mp2_template.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP3_Template ${TEMPLATE}/mp3_template.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP4_Template ${TEMPLATE}/mp4_template.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP5_Template ${TEMPLATE}/mp5_template.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP6_Template ${TEMPLATE}/mp6_template.cu ${wbhdr} ${wbsrc})


# Add Implementation Code
CUDA_ADD_EXECUTABLE(MP0 ${IMPLEMENT}/mp0.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP1 ${IMPLEMENT}/mp1.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP2 ${IMPLEMENT}/mp2.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP3 ${IMPLEMENT}/mp3.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP5_Implement ${IMPLEMENT}/mp4.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP6_Implement ${IMPLEMENT}/mp5.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP7_Implement ${IMPLEMENT}/mp6.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP8_Implement ${IMPLEMENT}/mp7.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP8_Implement ${IMPLEMENT}/mp8.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP1_ref ${REFERENCE}/mp1_implement.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP2_ref ${REFERENCE}/mp2_implement.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP3_ref ${REFERENCE}/mp3_implement.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP4_ref ${REFERENCE}/mp4_implement.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP5_ref ${REFERENCE}/mp5_implement.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP6_ref ${REFERENCE}/mp6_implement.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP7_ref ${REFERENCE}/mp7_implement.cu ${wbhdr} ${wbsrc})
# CUDA_ADD_EXECUTABLE(MP8_ref ${REFERENCE}/mp8_implement.cu ${wbhdr} ${wbsrc})
#Add Tools
CUDA_ADD_EXECUTABLE(MP1_Dataset tools/mp1_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP2_Dataset tools/mp2_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP3_Dataset tools/mp3_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP4_Dataset tools/mp4_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP5_Dataset tools/mp5_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP6_Dataset tools/mp6_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP7_Dataset tools/mp7_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(Bank_Conflict tools/bank_conflict.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(Memory_Transfer tools/memory_transfer.cu ${wbhdr} ${wbsrc})
