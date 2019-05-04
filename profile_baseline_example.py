import logging
import profile_utils

from solvers import solver_baseline

def run_baseline_example():
    logging.info('RUNNING SERIAL UNOPTIMIZED SOLVER')

    #solve the system
    return solver_baseline.solve(sample_bus_count, sample_constr_count)

if __name__ == '__main__':
    logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    #run some sample from the dataset
    sample_bus_count = 189
    sample_constr_count = 5

    profile_utils.profile_funcalls(run_baseline_example)
    profile_utils.viz_funcalls()

    profile_utils.profile_callgraph(run_baseline_example)
    profile_utils.profile_memory(run_baseline_example)
    profile_utils.profile_time(run_baseline_example)
    profile_utils.check_residuals(run_baseline_example, sample_bus_count, sample_constr_count)