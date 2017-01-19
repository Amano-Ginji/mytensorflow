#!/bin/bash

find . -name "*.h" -o -name "*.c" -o -name "*.cc" -o -name "*.cpp" -o -name "*.mk" -o -name "*.py" > cscope.files
cscope -bkq -i cscope.files
ctags -R
