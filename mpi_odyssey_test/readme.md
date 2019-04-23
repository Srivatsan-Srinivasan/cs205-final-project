### HOW TO RUN THIS ON ODYSSEY

1. Create a virtual environment CS205 in Odyssey (Since you can install packages only in that env.)
conda create -n CS205 python=3.6

2. Move to that virtual env
source activate CS205

3. Install mpi4py : conda install mpi4py

4. Run bash script (which calls the python file, specifies env. to run from, specifies number of partitions etc.) 
sbatch run.sbatch

Note : Check which partition you are in. The current code uses doshi-velez(my lab's partition). You will not have access to it.

Some useful odyssey links that I found 

1. Creating new env - https://www.rc.fas.harvard.edu/resources/documentation/software-on-odyssey/python/
2. mpi4py on odyssey - https://www.rc.fas.harvard.edu/resources/documentation/software-development-on-odyssey/mpi-for-python-mpi4py-on-odyssey/
3. Some general MPI intro - https://www.rc.fas.harvard.edu/resources/documentation/software-development-on-odyssey/mpi-software-on-odyssey/
4. Odyssey how to run jobs primer - https://www.rc.fas.harvard.edu/resources/running-jobs/
5. Partitions list and monitoring job in queue - https://www.rc.fas.harvard.edu/resources/running-jobs/#Slurm_partitions

(will add more as I come across)

PS : When you want to come out of the env. just do source deactivate CS205


