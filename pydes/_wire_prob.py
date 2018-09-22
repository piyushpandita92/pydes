"""
functions for the wire problem
"""


__all__ = ['read_output',
            'get_parallel_data',
            'write_input_file',
            'compute_samp_avg'
		   ]


def write_input_file(in_file,x,samples,out_file_name):
    """
    writes the input parallel code using mpi4py.
    """
    with open(in_file,'w') as fd:
        fd.write("""   
import os
import sys
import copy
import numpy as np
import math

#from _function_evaluation_wrapper import FunctionEvaluationWrapper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
#sys.path.append("/home/ppandit/code/pydes/code/")
import fem_wire_problem
from mpi4py import MPI as mpi

rank = mpi.COMM_WORLD.Get_rank()
size = mpi.COMM_WORLD.Get_size()
# Print info
if rank == 0:       # This guy is known as the root
    print '=' * 80
    print 'Collecting data for the wire problem in parallel'.center(80)
    print '=' * 80
    print 'Number of processors available:', size
# Wait for the root to print this message
mpi.COMM_WORLD.barrier()   
my_num_samples = {0:d}/size
mpi.COMM_WORLD.barrier()
if rank == 0:
    X = [{1:1.5f},{2:1.5f},{3:1.5f},{4:1.5f},{5:1.5f},{6:1.5f},{7:1.5f},{8:1.5f},{9:1.5f},{10:1.5f},{11:1.5f},{12:1.5f},{13:1.5f},{14:1.5f},{15:1.5f}]
else:
    X = None
X = mpi.COMM_WORLD.bcast(X)
my_X = X
wrapper = fem_wire_problem.FunctionEvaluationWrapper(0.05,0.1,1)
# This loop is actually completely independent
my_Y = []
for j in xrange(my_num_samples):
    if rank == 0:
        print 'sample ' + str((j + 1) * size).zfill(6)
    my_Y.append(wrapper(my_X))
my_Y = np.array(my_Y)
# All these outputs need to be sent to the root
all_Y = mpi.COMM_WORLD.gather(my_Y)
if rank == 0:
    y = np.vstack(all_Y)
    y = np.mean(y,axis=0)
    np.save('{16:s}', y)
""".format(samples,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],out_file_name))

def read_output(out_file):
    """
    reads the output stored in a particular array in the temporary directory.
    """
    #output_file = out_file +'.npy'
    y = np.load(out_file)
    return y

def compute_samp_avg(x,samples=20):
    """
    This method writes the file, runs the parallel code, reads the output.
    """
    tmp_dir = tempfile.mkdtemp()
    file_prefix  = os.path.join(tmp_dir,'collect_data_final')
    in_file = file_prefix + '.py'
    out_file_name = file_prefix + '.npy'
    write_input_file(in_file,x=x,samples=samples,out_file_name=out_file_name)
    cmd = ['mpiexec', '-n', '20', 'python', str(in_file)]
    #DEVNULL = open(os.devnull, 'wb')
    p = subprocess.Popen(cmd, cwd=tmp_dir)
    p.wait()
    assert p.returncode == 0
    #DEVNULL.close()
    #out_file = file_prefix + '.npy'
    out = read_output(out_file_name)
    shutil.rmtree(tmp_dir)
    return out

def get_parallel_data(x,samples,obj_true):
    """
    Computes the data in parallel for the final stage.
    """
    import os
    import sys
    import tempfile
    import subprocess
    import shutil
    #from _function_evaluation_wrapper import FunctionEvaluationWrapper
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import fem_wire_problem
    from mpi4py import MPI as mpi
    n = x.shape[0]
    Y = []
    for i in xrange(n):
        y = compute_samp_avg(x[i,:],samples)
        Y.append(y)
    return Y