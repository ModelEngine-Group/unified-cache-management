gcc -O2 -Wall -o test test.cc ../compress_lib/*.cc
./test BF16.bin
rm test