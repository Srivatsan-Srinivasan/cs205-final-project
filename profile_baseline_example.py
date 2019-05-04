import time
import pstats
import pickle
import logging
import cProfile
import logging.config

from pycallgraph import PyCallGraph
from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph import GlobbingFilter
from pycallgraph.output import GraphvizOutput
from memory_profiler import memory_usage
from baseline_example import run_baseline_example
from solvers.utils import calculate_residuals
from solvers.utils import get_filepaths
from solvers.utils import load_filepaths
from solvers.solver_utils import construct_NewtonBDMatrix
from solvers import solver_baseline

def profile_funcalls(fn):
    logging.info('PROFILING FUNCTION CALLS ...')

    cProfile.runctx('fn()', globals(), locals(), 'profile_example.out')

    logging.info('PROFILED FUNCTION CALLS.')

def viz_funcalls():
    p = pstats.Stats('profile_example.out')

    #plain sorted calls
    p.strip_dirs().sort_stats(-1).print_stats()

    #top 10 by cumulative time
    p.strip_dirs().sort_stats('cumulative').print_stats(10)

    #top 10 by call count
    p.strip_dirs().sort_stats('calls').print_stats(10)

    #top 10 by file name
    p.strip_dirs().sort_stats('filename').print_stats(10)

    #top 10 by primitive call count
    p.strip_dirs().sort_stats('pcalls').print_stats(10)
    
    #top 10 by total internal time
    p.strip_dirs().sort_stats('tottime').print_stats(10)


def profile_callgraph(fn):

    logging.info('PROFILING CALL GRAPH ...')

    config = Config()
    config.trace_filter = GlobbingFilter(exclude=[
        'ModuleSpec.*',
        'logging.*',
        '*_handle_fromlist*',
        'multiprocessing.*',
        'threading.*'
    ])

    graphviz = GraphvizOutput(output_file='profile_example.png')

    with PyCallGraph(output=graphviz, config=config):
        fn()

    logging.info('PROFILED CALL GRAPH.')

def profile_memory(fn):

    logging.info('PROFILING MEMORY ...')

    mem_usage = memory_usage(fn, interval=.01)

    logging.info('PROFILED MEMORY.')

    logging.info('max memory={:3f}MB.'.format(max(mem_usage)))
    pickle.dump(mem_usage, open('memory_example.out', 'wb'))

def profile_time(fn):

    logging.info('PROFILING TIME ...')

    start_time = time.time()

    fn()

    end_time = time.time()

    logging.info('PROFILED TIME.')

    logging.info('total time={:3f}s.'.format(end_time - start_time))
    pickle.dump(end_time - start_time, open('time_example.out', 'wb'))

def check_residuals(fn):
    
    #calculate the residuals for proposed solution
    logging.info('calculating residuals ...')
    fnames = get_filepaths(sample_bus_count, sample_constr_count, 'Data')
    model_data, rhs_data, y0_data, constr_data = load_filepaths(*fnames)
    newton_matrix = construct_NewtonBDMatrix(model_data, y0_data, constr_data)

    soln = fn()
    residuals = calculate_residuals(newton_matrix, rhs_data, fn())
    logging.info('finished calculating residuals.')

    #report residuals
    logging.info('residue={:3f}.'.format(residuals))
    pickle.dump(residuals, open('residuals_example.out', 'wb'))

def run_baseline_example():

    logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('RUNNING SERIAL UNOPTIMIZED SOLVER')

    #solve the system
    return solver_baseline.solve(sample_bus_count, sample_constr_count)

if __name__ == '__main__':
    logging.basicConfig(format = "%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    #run some sample from the dataset
    sample_bus_count = 189
    sample_constr_count = 5

    profile_funcalls(run_baseline_example)
    viz_funcalls()

    profile_callgraph(run_baseline_example)
    profile_memory(run_baseline_example)
    profile_time(run_baseline_example)
    check_residuals(run_baseline_example)