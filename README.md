# CS205 Spring 2019, Final Project
#### Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru

* Proposal Google Slides: https://docs.google.com/presentation/d/1oGa-gHbte9kbHfcSGOF94_0M-nd7p2r2Ky5b_dz2hU8/edit?usp=sharing
* Design Overleaf Slides: https://www.overleaf.com/6991635993rpjtwjzbgmfg
* Project Report/Website: https://srivatsan-srinivasan.github.io/cs205-final-project/

# Compute Environment

All codes were profiled on Harvard's Odyssey Compute Infrastructure.

## Environment Setup
To setup the environment on Odyssey, do exactly the follows. It'll setup the appropriate python version, packages for running and profiling the solvers. 

\$ `module load Anaconda3/5.0.1-fasrc02`

\$ `conda create -n cs205_env python=3.6`

\$ `source activate cs205_env`

\$ `conda install numpy scipy matplotlib graphviz python-graphviz memory_profiler`

\$ `pip install pycallgraph`


## Summary of Directory Structure

`profiler_utils.py`

`profile_some_solver_example.py`

`some_solver_example.py`

`\solvers\some_solver_implementation.py`

`\solvers\solver_utils.py`

`\solvers\utils.py`

Please refer to the individual python source files for more detailed documentation and program flow. All functions have decent docstrings.

Implementations of the various versions of the solvers fall into the pattern `\solvers\some_solver_implementation.py`. For example, for the direct solve, refer to `\solvers\solver_baseline.py` or for the fully parallel multigrid implementation, refer to `\solvers\solver_multilevel.py`. Note that the major of the common logic for these solvers has been abstracted out to the two utility files in the directory `\solvers\solver_utils.py` for solver specific utilities (actual implementation of multigrid algorithm, etc.) and `\solvers\utils.py` for very generic utilities (for loading appropriate files given number of busses, contengencies, etc.).

These newton step solvers can be involed by running either `some_solver_example.py` to just run the solver or to profile the solver, a user can change the profilers being used and run `profile_some_solver_example.py`. For example, for the baseline profiler, see `profile_baseline_example.py` and for the profiler for the fully parallelized multigrid, see `profile_multilevel_example.py`. Please note that for the fully parallelized version, we have a more limited set of tested profilers (only time and residual profilers) because of complications with mpirun/mpiexec. These driver scripts expect user arguments for the characteristics of the power network to run on, parallelization options, etc. which will be discussed below. Overally, the typical arrangement of the code in terms of the actual call graph is visualized below.

![alt text](https://i.imgur.com/2sL7tws.png)

This call graph is an example of one of the profilers that we offer for serial codes! It was used heavily to inform our choice of parallelization.

## Actually running the code.

To actually run the profilers, once the environment is setup properly, do the follows. Again to reiterate, note that the user has to provide system arguments to the python scripts that determine the characteristics of the power network and the parallelization level that is being used in the solvers.

Again, please refer to the python files which have decent docstrings on the ordering of the input arguments. Listed below are a subset of some combinations that you could try, from the root directory:

Please note that sometimes mpirun/mpiexec and python results in strange interactions for print statements. When this is the case, the print statements can be flushed by replacing `python something.py` with `python -u something.py`.

`python baseline_example.py 189 3` runs the baseline solver (direct solve with scipy for a bus size of 189 with 3 contingencies.

`python serial_multilevel_example.py 189 3 false 0` runs the serial version of the multigrid solver without parallelism at block 1 (generation of the newton matrix) with a bus size of 189 and 3 contingencies.

`python serial_multilevel_example.py 189 3 true 0` runs the serial version of the multigrid solver with parallelism at block 1 (generation of the newton matrix) with a bus size of 189 and 3 contingencies.

`python serial_multilevel_example.py 189 3 true 2` runs the serial version of the multigrid solver with parallelism at block 1 using 2 processors in PyMP (generation of the newton matrix) with a bus size of 189 and 3 contingencies.

`python multilevel_example.py 189 4 true true 0` runs the parallel version of the multigrid solver with parallelism at block 1, 2, 3 with a bus size of 189 and 4 contingencies.

`python multilevel_example.py 189 4 false true 0` runs the parallel version of the multigrid solver with parallelism at block 1, 2 with a bus size of 189 and 4 contingencies.

`python multilevel_example.py 189 4 true false 0` runs the parallel version of the multigrid solver with parallelism at block 2, 3 with a bus size of 189 and 4 contingencies.

Additionally, we provide some implementation of profilers to be used. All of the appropriate profilers work with the serial codes. Only the time and the parallel specific residual profilers work with the parallelized codes. All the working profilers have been enabled by default. Be careful of runtimes here, the code is re-run once everytime for each profiler.

`python python profile_baseline_example.py 189 3` runs all the profilers for the baseline example (function call stats, call graph vizualization, memory usage, timing, residuals).

`python profile_serial_multilevel_example.py 189 3 false 0` runs all profilers for the completely serial multilevel solver.

`python profile_serial_multilevel_example.py 189 3 true 0` runs all profilers for the serial multilevel solver with parallelism only at block 1. NOTE THAT THIS RUNS BUT THE OUTPUT FORM THE PROFILERS HAVE NOT BEEN VERIFIED. WE EXPECT THIS MIGHT VERY WELL BE BROKEN DUE TO INTERACTION WITH PyMP. We have however tested the timing profiler and the residual profilers, which are the most important to our cause.

`python profile_multilevel_example.py 189 3 true true 0` runs all profilers for the fully parallelized multilevel solver. Note that only the timing and residual profilers have been verified for the fully parallelized implementations.


## Reproducing experiments

All our experiments had a setup as follows :  the PyMP based parallelization used K+1 cores(2,3,....7)  within  one  compute  node  in  Odyssey  and  the  mpi4py  parallelization  used  K+2compute nodes (base, K contingencies, power injection) in Odyssey while each node almost effectively (except for a few implicit multi-core operations of some scipy modules) used only one core per node (despite the available number of cores ranging from 8-32 in each node). For our experiments, we profiled our solvers for K=1,2...6.

For example, to repeat the experiments for the fullest extent of parallelization for the largest problem:

`srun -p shared --pty --ntasks=8 --ntasks-per-node=1 --cpus-per-task=7 --mem-per-cpu=1500 -t 0-00:30  /bin/bash` to request the appropriate resources (note that we always restrict the total amount of memory per node to 12GB). Notice the use of 7 (K+1) cores per node and 8 (K+2) nodes for our expected problem size of K=8 contengencies with 2224 busses.

To run the profiler to get the execution time, execute the profiler script:

`python profile_multilevel_example.py 2224 6 true true 0`


