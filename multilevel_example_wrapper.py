import sys
from solvers import solver_multilevel

def run_multilevel_example(bus_count, constr_count):
	soln = solver_multilevel.solve(bus_count, constr_count)
	if soln is not None:
		#worker nodes return None, master node returns the solutions
		return soln

if __name__ == '__main__':
	run_multilevel_example(int(sys.argv[1]), int(sys.argv[2]))