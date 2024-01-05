# Dispersion tests (mintrop)

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

from dispersion_tfourier import *
from multiprocessing import Pool

# degree=np.arange(3,31,2)
degree=np.arange(3,31)

pool = Pool(processes=40) # start 63 worker processes

methods=np.array(['RK7','RK2','2MS','RK4','KRY','FA','HORK'])
Ndt=np.arange(1,31)
degree=np.arange(3,31)
# frequencies=np.array([10,15,20,25,30])
frequencies=np.array([15])
# degree=np.array([15])
delta=0.0
# ---------------------------------------------------------------------------------------------------------
# Computing the solution in the dispersion model
# ---------------------------------------------------------------------------------------------------------
for k in frequencies:
    pool.apply_async(solution_dispersion_cluster, args=("RK7",1,np.array([1]),0.0025,2,k,3,delta,))
    for i in Ndt:
        print("hola-",i)
        for j in range(4):
            pool.apply_async(solution_dispersion_cluster, args=(methods[j],0,np.array([i]),0.01,1,k,3,delta,))
        for j in range(3):
            for l in degree:
                pool.apply_async(solution_dispersion_cluster, args=(methods[j+4],np.array([l]),np.array([i]),0.01,1,k,3,delta,))
pool.close()
pool.join()
