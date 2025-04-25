# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

project("mnn_vits")

add_library(${CMAKE_PROJECT_NAME} SHARED
        mnn_vits_jni.cpp
        mnn_vits_loader.cpp
)
#mnn
set (MNN_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/MNN")

set(MNN_USE_LOGCAT ON)
option(MNN_USE_LOGCAT "Use Logcat intead of print for info" ON)

message(ERROR "MNN_ROOT: ${MNN_ROOT}")
include_directories("${MNN_ROOT}/include/")

set(LIB_PATH "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")

# Link the shared library
add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN.so")

add_library(mnn_Express SHARED IMPORTED)
set_target_properties(mnn_Express PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN_Express.so")


target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        android
        log
        MNN
        mnn_Express
)
