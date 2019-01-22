#!/bin/bash
export PFILE=_results_temp/profile.pstats
python -m cProfile -o ${PFILE} ./quadrotor.py -q random -r -trj 40 &&
snakeviz ${PFILE}