import sys
from profile_utils import check_residuals_static
from solvers import solver_multilevel

def run_multilevel_example_residuals(bus_count, constr_count):
	soln = solver_multilevel.solve(bus_count, constr_count)
	if soln is not None:
		#worker nodes return None, master node returns the solutions
		check_residuals_static(soln, bus_count, constr_count)

if __name__ == '__main__':
	run_multilevel_example_residuals(int(sys.argv[1]), int(sys.argv[2]))