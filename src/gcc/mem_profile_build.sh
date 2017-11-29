ARGS=
#ARGS=-m32
COMPILER=clang++-3.6
COMPILER=clang++-3.8
#COMPILER=g++

$COMPILER $ARGS  -O2 -o mem_profilee -I.. -std=c++11 -Wall -Wextra -Wno-unused-parameter mem_profile.cpp
#$COMPILER $ARGS   -g    -I.. -std=c++11 -Wall -Wextra -Wno-unused-parameter mem_profile.cpp 
