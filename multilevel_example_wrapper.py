import sys
from solvers import solver_multilevel

def run_multilevel_example(bus_count, constr_count, fullParallel, newtonParallel, newton_nproc):
	soln = solver_multilevel.solve(bus_count, constr_count, fullParallel, newtonParallel, newton_nproc)
	if soln is not None:
		#worker nodes return None, master node returns the solutions
		return soln

if __name__ == '__main__':
	sys.argv[3] = sys.argv[3].lower() == 'true'
	sys.argv[4] = sys.argv[4].lower() == 'true'
	run_multilevel_example(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]))