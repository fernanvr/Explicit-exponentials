# Corner Model (mintrop)

from Testes import *
from multiprocessing import Pool

# degree=np.arange(3,31,2)
degree=np.arange(3,31)

pool = Pool(processes=30) # start 63 worker processes

T=2.2
T_frac_snapshot=1/2
methods=np.array(['RK7','RK2','RK4','2MS','FA','HORK','KRY'])

pool.apply_async(wave_eq, args=(0.005,'scalar_dx2',2,1,1.0,30,'8',T,T_frac_snapshot,np.array([1]),0,'2D_heterogeneous_3c','RK7','H_amplified',))

for l in range(50):
    for i in range(4):
        pool.apply_async(wave_eq, args=(0.01,'scalar_dx2',2,1,1.0,30,'8',T,T_frac_snapshot,np.array([l+1]),0,'2D_heterogeneous_3b',methods[i],'H_amplified',))

    for i in range(len(degree)):
        for j in range(3):
            pool.apply_async(wave_eq, args=(0.01,'scalar_dx2',2,1,1.0,30,'8',T,T_frac_snapshot,np.array([l+1]),np.array([degree[i]]),'2D_heterogeneous_3b',methods[4+j],'H_amplified',))
        # if degree[i]<15 or (l>=5)*(degree[i]<20) or (l>=7)*(degree[i]<25) or l>=9:
        #     for j in range(3):
        #             pool.apply_async(wave_eq, args=(0.01,'scalar_dx2',2,1,1.0,30,'8',T,T_frac_snapshot,np.array([l+1]),np.array([degree[i]]),'2D_heterogeneous_3b',methods[4+j],'H_amplified',))
pool.close()
pool.join()