'''
PARALLEL MULTILEVEL SOLVER

Parallel Codes for Multi-Level Algebraic Solver for Parallelized Newton Step 
in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''

import sys
from profile_utils import check_residuals_static
from solvers import solver_multilevel

def run_multilevel_example_residuals(bus_count, constr_count, fullParallel, newtonParallel, newton_nproc):
	'''
	wrapper fn that helps to make easier the profiling of the residuals for the multi level solver. 

	Args:
		bus_count (int): the number of busses in the power network
		costr_count (int): the number of constraints in the power network
		fullParallel (bool): parallelize block 3 as well as block 2 in the solver ? (see report/website)
		newtonParallel (bool): parallelize generation of the newton step matrix?
		newton_nproc (int): number of procs to use in parallelization of newton step matrix. 0 defaults to full parallization

	Returns:
		Nothing. This function just executes the solvers, used specifically for calculating the residuals from the solver.
	'''

	#runs the parallelized solver given bus count, constraint count and options regarding extend of parallelization	
	soln = solver_multilevel.solve(bus_count, constr_count, fullParallel, newtonParallel, newton_nproc)
	if soln is not None:
		#worker nodes return None, master node returns the solutions
		check_residuals_static(soln, bus_count, constr_count)

if __name__ == '__main__':
	sys.argv[3] = sys.argv[3].lower() == 'true'
	sys.argv[4] = sys.argv[4].lower() == 'true'
	run_multilevel_example(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]))