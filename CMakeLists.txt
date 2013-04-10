cmake_minimum_required(VERSION 2.8)

project(online_adaboost)
find_package(OpenCV REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(
  ${PROJECT_SOURCE_DIR}/lib
  ${OpenCV_LIB_DIR}
)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_library(adaboost 	src/Selector.cpp
			src/StrongClassifier.cpp)

target_link_libraries(adaboost ${OpenCV_LIBS})

add_executable(demo_adaboost  src/demo_adaboost.cpp )
target_link_libraries(demo_adaboost adaboost ${OpenCV_LIBS})


