#!/usr/bin.env/python

from mpi4py import MPI
nproc = MPI.COMM_WORLD.Get_size()
iproc = MPI.COMM_WORLD.Get_rank()
inode = MPI.Get_processor_name()

if iproc == 0: print ("This code is a test for mpi4py")

for i in range(0,nproc):
	MPI.COMM_WORLD.Barrier()
	if iproc == i :
		print('Rank %d out of %d' % (iproc,nproc))

MPI.Finalize()
