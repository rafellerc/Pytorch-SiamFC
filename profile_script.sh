#!/bin/bash
# This script wraps the python cProfiler, it automatically profiles the given
# script, and generates the graph representing it's execution. It saves the output
# of the profiler to 'profile.pstats' and uses it to create a graphical representation
# of the program's execution using gprof2dot, which saves the graph in the
# file callgraph.svg, which can be viewed to analyse the time profile of the
# program.

# This script can be called with the alias (actually a function) 'profile'.
# This alias is defined in the ./bashrc

# Example:
# $ profile <your_script.py> <your_arguments>
# $ profile train.py -w 0

# Otherwise execute it as:
# $ ./profile_script.sh train.py -w 0

echo "Executing " $1 " with cProfile."
{
CUDA_LAUNCH_BLOCKING=1 python3 -m cProfile -o profile.pstats "$@" &&
echo "Execution successful, saved stats to 'profile.pstats'" &&
# Requires 'gprof2dot' to be in home directory. To get it type:
# git clone https://github.com/jrfonseca/gprof2dot
# in the home directory.
~/gprof2dot/gprof2dot.py -f pstats profile.pstats | dot -Tsvg -o callgraph.svg &&
echo "Saved graph of execution to callgraph.svg"
} || {
echo "Failed to profile" $1
}