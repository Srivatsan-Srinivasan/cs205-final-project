'''
HELPER FUNCTION FOR PROFILING CODE

Helper Functions to Help in Profiling Newton Step Solvers in Security Constrained Optimal Power Flow

CS205 Spring 2019, Final Project
Authors: Aditya Karan, Srivatsan Srinivasan, Cory Williams, Manish Reddy Vuyyuru
'''

import time
import pstats
import pickle
import logging
import cProfile

from pycallgraph import PyCallGraph
from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph import GlobbingFilter
from pycallgraph.output import GraphvizOutput
from memory_profiler import memory_usage
from solvers.utils import calculate_residuals
from solvers.utils import get_filepaths
from solvers.utils import load_filepaths
from solvers.solver_utils import construct_NewtonBDMatrix

def profile_funcalls(fn):
    '''
    Given the function to profile the function calls of using cProfile, saves the log.

    Args:
        fn (python func): Function to profile

    Returns:
        Nothing (saves cprofile output to file)
    '''
    logging.info('PROFILING FUNCTION CALLS ...')

    cProfile.runctx('fn()', globals(), locals(), 'profile_example.out')

    logging.info('PROFILED FUNCTION CALLS.')

def viz_funcalls():
    '''
    Given that the cprofile output has been saved to a file, parses it to viz the more useful
    statistics.

    Args:
        Nothing (uses cprofile save file generated with profile_funcalls())

    Returns:
        Nothing (prints the most useful summaries from cprofile save file)
    '''
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
    '''
    Given the function to profile (serial function), profiles the python call graph for the function.

    Args:
        fn (python func): Function to profile
    Returns: 
         Nothing (saves a visualization of the call graph to file.) 
    '''
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
    '''
    Given the function to profile, profiles the memory usage for the function.

    Args:
        fn (python func): Function to profile
    Returns:
        Nothing (saves the memory profile of the function recorded at fixed intervals)
    '''

    logging.info('PROFILING MEMORY ...')

    mem_usage = memory_usage(fn, interval=.01)

    logging.info('PROFILED MEMORY.')

    logging.info('max memory={:3f}MB.'.format(max(mem_usage)))
    pickle.dump(mem_usage, open('memory_example.out', 'wb'))

def profile_time(fn):
    '''
    Given the function to profile, profiles the time taken for execution of the function.

    Args:
        fn (python func): Function to profile

    Returns:
        Nothing (saves the time for execution to a save file)
    '''

    logging.info('PROFILING TIME ...')

    start_time = time.time()

    fn()

    end_time = time.time()

    logging.info('PROFILED TIME.')

    logging.info('total time={:3f}s.'.format(end_time - start_time))
    pickle.dump(end_time - start_time, open('time_example.out', 'wb'))

def check_residuals(fn, sample_bus_count, sample_constr_count):
    '''
    Given the function to a newton step solver, the number of busses and constraints in the network,
    calculates the residuals for the function (a measure of the deviations from the expected answer).

    Args:
        fn (python func): Function to profile
        sample_bus_count (int): the number of busses in the power network
        sample_constr_count (int): the number of constraints in the power network

    Returns:
        Nothing (saves to file the result of the residue calculation)
    '''
    
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


def check_residuals_static(soln, sample_bus_count, sample_constr_count):
    '''
    Given the solution provided by a newton step solver, the number of busses and constraints in the network,
    calculates the residuals for the function (a measure of the deviations from the expected answer).

    Args:
        soln (scipy.sparse or numpy matrix): the solution provided by a newton step solver
        sample_bus_count (int): the number of busses in the power network
        sample_constr_count (int): the number of constraints in the power network

    Returns:
        Nothing (saves to file the result of the residue calculation)
    '''

    #calculate the residuals for proposed solution
    logging.info('calculating residuals ...')
    fnames = get_filepaths(sample_bus_count, sample_constr_count, 'Data')
    model_data, rhs_data, y0_data, constr_data = load_filepaths(*fnames)
    newton_matrix = construct_NewtonBDMatrix(model_data, y0_data, constr_data)

    residuals = calculate_residuals(newton_matrix, rhs_data, soln)
    logging.info('finished calculating residuals.')

    #report residuals
    logging.info('residue={:3f}.'.format(residuals))
    #temporary hack around
    print('residue={:3f}.'.format(residuals))
    pickle.dump(residuals, open('residuals_example.out', 'wb'))