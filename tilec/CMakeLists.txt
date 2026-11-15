cmake_minimum_required(VERSION 3.16)
project(tilec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TILEC_SANITIZE "Build with AddressSanitizer + UBSan" OFF)
option(TILEC_WERROR "Treat warnings as errors" ON)

add_library(tilec_core
  src/Lexer.cpp
  src/Parser.cpp
  src/IRGen.cpp
  src/IR.cpp
  src/Passes.cpp
  src/Fusion.cpp
  src/Interpreter.cpp
  src/LLVMCodegen.cpp)
target_include_directories(tilec_core PUBLIC include)
target_compile_options(tilec_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(tilec tools/tilec/main.cpp)
target_link_libraries(tilec PRIVATE tilec_core)
target_compile_options(tilec PRIVATE -Wall -Wextra -Wpedantic)

if(TILEC_WERROR)
  target_compile_options(tilec_core PRIVATE -Werror)
  target_compile_options(tilec PRIVATE -Werror)
endif()

if(TILEC_SANITIZE)
  foreach(t tilec_core tilec)
    target_compile_options(${t} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(${t} PRIVATE -fsanitize=address,undefined)
  endforeach()
endif()

enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME lit
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/run_lit.py
          --tilec $<TARGET_FILE:tilec> ${CMAKE_SOURCE_DIR}/tests/lit)
add_test(NAME e2e
  COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_SOURCE_DIR}/tests/e2e)
set_tests_properties(e2e PROPERTIES ENVIRONMENT "TILEC=$<TARGET_FILE:tilec>")
add_test(NAME fuzz
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/fuzz/fuzz.py
          --tilec $<TARGET_FILE:tilec> --iters 150 --seed 1)
