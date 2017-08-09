ARGS=
ARGS=-m32
COMPILER=clang++-3.6
COMPILER=clang++-3.8
COMPILER=g++

$COMPILER $ARGS -g -O2 -I.. -std=c++11 -Wall -Wextra -Wno-unused-parameter standalone.cpp 
