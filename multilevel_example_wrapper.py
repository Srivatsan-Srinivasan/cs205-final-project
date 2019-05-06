import sys
from solvers import solver_multilevel
import time
def run_multilevel_example(bus_count, constr_count):
        start_time = time.time()
        soln = solver_multilevel.solve(bus_count, constr_count, fullParallel = True)
        if soln is not None:
                print("Time diff is %s" % str(time.time() - start_time))
                print("%s" % (str(soln)))
		#worker nodes return None, master node returns the solutions
                return soln

if __name__ == '__main__':
                run_multilevel_example(int(sys.argv[1]), int(sys.argv[2]))
