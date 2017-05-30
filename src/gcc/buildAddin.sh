#Builds (at least on Ubuntu 16.04LTS) an python extension
#iisignature.so which can be loaded into python with
#    import iisignature
#For full functionality, copy the iisignature_data directory
#into your working directory.

#There's no optimization flag here like -O2, nor -g

ARGS=

#uncomment one of these
#COMPILER=clang++-3.6 ; ARGS=$ARGS\ -Wno-c++14-extensions
COMPILER=clang++-3.8
#COMPILER=g++

#uncomment one of these
#outputfile=iisignature_localbuild.so
OUTPUTFILE=iisignature.so

#uncomment one of these
PYTHON=python2.7
#PYTHON=python3.5 ; ARGS=$ARGS\ -L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu

$COMPILER $ARGS -std=gnu++11 "-DVERSION=\"$COMPILER standalone build\"" -Wall -Wextra -Wno-unused-parameter -Werror -I/usr/include/$PYTHON -l$PYTHON -shared -fPIC ../pythonsigs.cpp -o $OUTPUTFILE
