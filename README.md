# CS205 Spring 2019, Final Project
# Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru

* Proposal Google Slides: https://docs.google.com/presentation/d/1oGa-gHbte9kbHfcSGOF94_0M-nd7p2r2Ky5b_dz2hU8/edit?usp=sharing
* Design Overleaf Slides: https://www.overleaf.com/6991635993rpjtwjzbgmfg
* Project Report/Website: https://srivatsan-srinivasan.github.io/cs205-final-project/

# Compute Environment

All codes were profiled on Harvard's Odyssey Compute Infrastructure.

## Environment Setup
To setup the appropriate environment 




![alt text](https://github.com/Srivatsan-Srinivasan/cs205-final-project/blob/master/imgs/profile_example.png)

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

### INSTRUCTED DEPRECATED SEE BELOW FOR SHORT SUMMARY OF NEW

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

--ntasks=1 --ntasks-per-node=1 --cpus-per-task=1 --mem-per-cpu 16GB   
--ntasks=1 --ntasks-per-node=1 --cpus-per-task=8 --mem-per-cpu 16GB   
--ntasks=1 --ntasks-per-node=1 --cpus-per-task=32 --mem-per-cpu 16GB 

--ntasks=8 --ntasks-per-node=1 --cpus-per-task=1 --mem-per-cpu 12GB  
srun -p test --pty --ntasks=8 --ntasks-per-node=1 --cpus-per-task=1 -t 0-01:00 --mem-per-cpu=12000 /bin/bash

--ntasks=8 --ntasks-per-node=1 --cpus-per-task=8 --mem-per-cpu 2GB


--ntasks=8 --ntasks-per-node=1 --cpus-per-task=4

### NEW runners

#### to run solver
python baseline_example.py
python serial_multilevel_example.py

#### to profiler solver
python profile_baseline_example.py
python profile_serial_multilevel_example.py
