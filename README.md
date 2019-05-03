# cs205-final-project
This repository will be used to commit code necessary for the final project of CS205 course. Contributors are

1. Srivatsan Srinivasan

2. Manish Reddy Vuyyuru 

3. Aditya Karan 

4. Cory Williams

* Google slides for the proposal : https://docs.google.com/presentation/d/1oGa-gHbte9kbHfcSGOF94_0M-nd7p2r2Ky5b_dz2hU8/edit?usp=sharing

* Overleaf link for the design presentation : https://www.overleaf.com/4973698375xwjqbstbnfxp


# Compute Environment

Odyssey Compute Cluster

## Odyssey Modules

## Python Packages

## Environment Setup

$ module load Anaconda3/5.0.1-fasrc02
$ conda create -n cs205_env python=3.6
$ source activate cs205_env
$ conda install numpy scipy matplotlib graphviz python-graphviz memory_profiler
$ pip install pycallgraph

## Run

##### simple parallel example 2224 bus, 2 constraints (4 node)
$ srun -p test --pty -n 1 -N 4 -t 0-01:00 --mem-per-cpu=4000 /bin/bash
$ module load Anaconda3/5.0.1-fasrc02
$ source activate cs205_env
$ mpirun -n 4 python -u newton.py

expected timing: ~ 184.2681279182434s

##### simple serial example 2224 bus, 2 cosntraints (1 node)
$ srun -p test --pty -n 1 -N 1 -t 0-01:00 --mem-per-cpu=4000 /bin/bash
$ module load Anaconda3/5.0.1-fasrc02
$ source activate cs205_env
$ python newton_serial.py

expected timing: ~ 193.45466351509094s