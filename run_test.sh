#!/bin/bash

# Ensure the binary is built
PROGRAM=./test_gemm  # Replace w>

# Loop from 1000 to 1000000 in steps (e.g., 1000, 10000, 100000, etc.)
for (( rows=512; rows<=1024; rows=rows*2)); do
    for (( columns=1000; columns<=1000; columns+=100 )); do
        for (( clusters=20; clusters<=100; clusters+=5 )); do
            $PROGRAM $rows $columns $clusters
        done
    done
done
