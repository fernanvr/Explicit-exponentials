import numpy as np
import dispersion_tfourier as dis
from multiprocessing import Pool

methods=np.array(['RK7','RK2','2MS','RK4','KRY','FA','HORK'])
Ndt=np.arange(1,31)
degree=np.arange(3,31)
# degree=np.arange(3,21)
# degree=np.array([15])
frequencies=np.array([10,15,20,25])
# frequencies=np.array([25])
# ---------------------------------------------------------------------------------------------------------
# Computing the solution in the dispersion model
# ---------------------------------------------------------------------------------------------------------
# for k in frequencies:
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#     print('k: ', k)
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#     dis.solution_dispersion("RK7",1,np.array([1]),dx=0.005,dx_factor=2,f0=k,test=3,replace=1)
#     for i in Ndt:
#         print('----------------------------------------------------------------------------------------------------------')
#         print('----------------------------------------------------------------------------------------------------------')
#         print('i: ',i)
#         print('----------------------------------------------------------------------------------------------------------')
#         print('----------------------------------------------------------------------------------------------------------')
#         for j in range(4):
#             dis.solution_dispersion(methods[j],0,np.array([i]),f0=k,dx=0.01,test=3,replace=1)
#         for j in range(3):
#             for l in degree:
#                 print('j,l',j,l)
#                 dis.solution_dispersion(methods[j+4],np.array([l]),np.array([i]),f0=k,dx=0.01,test=3,replace=1)


# ---------------------------------------------------------------------------------------------------------
# Error estimation for each method using the minimum tim-step for each frequency, to account for the spatial discretization impact
# ---------------------------------------------------------------------------------------------------------
# for f0 in frequencies:
#     dis.graph_estimate_diss_disp(methods[::-1],degree,Ndt,f0=f0,results='convergence1_dispersion_dissipation1',dx=0.01)

# ---------------------------------------------------------------------------------------------------------
# Error estimation of \Delta t max for each method such that the time error is less than 50% of the spatial error
# ---------------------------------------------------------------------------------------------------------
# for f0 in frequencies:
#     dis.graph_estimate_diss_disp_max_dt(methods[::-1],degree,Ndt,f0=f0,results='convergence1_dispersion_dissipation1',tol_mult=1.3,dx=0.01)


# ---------------------------------------------------------------------------------------------------------
# Error estimation of computational efficiency for each method such that the time error is less than 50% of the spatial error
# ---------------------------------------------------------------------------------------------------------
# for f0 in frequencies:
#     dis.graph_estimate_diss_disp_eff(methods[::-1],degree,Ndt,f0=f0,results='dispersion_dissipation1',dx=0.01)


# ---------------------------------------------------------------------------------------------------------
# Snapshoots graphics of the reference solution at time t=0.77 and T=1.25
# ---------------------------------------------------------------------------------------------------------

# dis.solution_dispersion("RK7",1,np.array([1]),dx=0.0025,dx_factor=4,f0=15.00001,fig_ind=1,replace=0,test=3)
# dis.solution_dispersion("RK7",1,np.array([1]),dx=0.0025,dx_factor=4,f0=15,fig_ind=1,replace=0,test=3)



# dis.solution_dispersion("RK7",np.array([15]),np.array([1]),dx=0.005,dx_factor=2,f0=15,fig_ind=1,replace=0,test=3)

# dis.graph_estimate_diss_disp(methods[::-1],degree,Ndt,f0=15,results='convergence2',dx_factor=2,dx=0.01,test=3)
# dis.graph_estimate_diss_disp_max_dt(methods[::-1],degree,Ndt,f0=15,results='convergence2',dx_factor=2,dx=0.01,test=3)


# dis.graph_estimate_diss_disp(methods[::-1],degree,Ndt,f0=20,results='convergence1_dispersion_dissipation1',dx_factor=4,dx=0.01,test=3)

# dis.graph_estimate_diss_disp_max_dt(methods[::-1],degree,Ndt,f0=15,results='dispersion',tol_mult=1.3,dx=0.01)

# for l in degree:
#     print('l',l)
#     dis.solution_dispersion(methods[4],np.array([l]),np.array([12]),f0=15,test=101)


# dis.solution_dispersion("RK7",1,np.array([1]),dx=0.00125,dx_factor=4,fig_ind=1,replace=0)
# dis.solution_dispersion("KRY",np.array([30]),np.array([12]),fig_ind=1,replace=0)
# dis.solution_dispersion("KRY",np.array([20]),np.array([12]),fig_ind=1,replace=0)
# dis.solution_dispersion("KRY",np.array([15]),np.array([12]),fig_ind=1,replace=0)
# dis.solution_dispersion("RK7",1,np.array([1]),dx=0.005,dx_factor=1,fig_ind=1,replace=0)


# dis.graph_methods_dt_max(methods,degree,Ndt)


# dis.graph_wave_disp_diss(method='KRY',degree=np.array([15]),Ndt=np.array([5]),f0=15,results='convergence',dx=0.01,dx_factor=2)
# dis.graph_wave_disp_diss(method='KRY',degree=np.array([25]),Ndt=np.array([10]),f0=15,results='convergence')
# dis.graph_wave_disp_diss(method='KRY',degree=np.array([30]),Ndt=np.array([10]),f0=15,results='convergence')

# dis.graph_wave_disp_diss(method='HORK',degree=np.array([20]),Ndt=np.array([3]))


Ndt=np.arange(1,31)
# dis.graph_estimate_diss_disp(np.array(['HORK']),degree,Ndt)
methods=np.array(['KRY','FA','HORK','RK7','RK2','2MS','RK4'])

# dis.graph_estimate_diss_disp(methods,degree,np.array([1]),f0=f0,results='convergence_dispersion_dissipation1')
# dis.graph_estimate_diss_disp(methods,degree,np.array([1]),f0=f0,results='dissipation2')
# dis.graph_estimate_diss_disp_max_dt(methods,degree,Ndt,f0=20,results='dissipation1')
# dis.graph_estimate_diss_disp_eff(methods,degree,Ndt,f0=20,results='dissipation1')

# for f0 in np.array([15]):
#     dis.graph_estimate_diss_disp(methods,degree,Ndt,f0=f0,results='convergence_dispersion_dissipation1')
    # dis.graph_estimate_diss_disp(methods,degree,Ndt,f0=f0,results='dissipation2')
# for f0 in np.array([10,15,20,25,30]):
#     # dis.graph_estimate_diss_disp_max_dt(methods,degree,Ndt,f0=f0,results='convergence_dispersion_dissipation1')
#     # dis.graph_estimate_diss_disp_max_dt(methods,degree,Ndt,f0=f0,results='dissipation2')
#     dis.graph_estimate_diss_disp_max_dt(methods,degree,Ndt,f0=f0,results='dissipation1')
# for f0 in np.array([10,15,20,25,30]):
#     # dis.graph_estimate_diss_disp_eff(methods,degree,Ndt,f0=f0,results='convergence_dispersion_dissipation1')
#     dis.graph_estimate_diss_disp_eff(methods,degree,Ndt,f0=f0,results='dispersion')
#     # dis.graph_estimate_diss_disp_eff(methods,degree,Ndt,f0=f0,results='dissipation2')





# experiments to understand dispersion and dissipation
# f0=20
# dx=0.005
# trans_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_1_ord_8_Ndt_1_dx_'+str(dx/4)+'_points_'+str(2)+'_f0_'+str(f0)+'_transform.npy')
#
# trans1=np.load('Dispersion_S/2MS_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(2)+'_f0_'+str(f0)+'_transform.npy')
# trans2=np.load('Dispersion_S/RK4_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(2)+'_f0_'+str(f0)+'_transform.npy')
#
# a1=trans_ref/trans1
# a1[(np.abs(trans_ref)+np.abs(trans1))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans1))])))]=1
# a1[((np.abs(trans_ref)+np.abs(trans1))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans1))]))))*(trans1==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans1))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans1))]))))*(trans1==0)]
#
# a2=trans_ref/trans2
# a2[(np.abs(trans_ref)+np.abs(trans2))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans2))])))]=1
# a2[((np.abs(trans_ref)+np.abs(trans2))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans2))]))))*(trans2==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans2))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans2))]))))*(trans2==0)]
#
# import matplotlib.pyplot as plt
# plt.plot(np.abs(trans_ref))
# plt.plot(np.abs(trans1))
# plt.plot(np.abs(trans2))
# plt.show()
#
# plt.plot(np.log(np.abs(trans_ref)))
# plt.plot(np.log(np.abs(trans1)))
# plt.plot(np.log(np.abs(trans2)))
# plt.show()
#
# plt.plot(np.abs(np.abs(trans_ref)-np.abs(trans1))*np.abs(np.log(np.abs(trans_ref))-np.log(np.abs(trans1))))
# plt.plot(np.abs(np.abs(trans_ref)-np.abs(trans2))*np.abs(np.log(np.abs(trans_ref))-np.log(np.abs(trans2))))
# plt.show()
#
#
# # plt.plot(np.abs(a1))
# # plt.plot(np.abs(a2))
# # plt.show()
#
# # plt.plot(np.angle(trans_ref))
# # plt.plot(np.angle(trans1))
# # plt.plot(np.angle(trans2))
# # plt.show()
# # plt.plot(np.abs(np.angle(np.exp(1j*(np.angle(trans_ref)-np.angle(trans1))))))
# # plt.plot(np.abs(np.angle(np.exp(1j*(np.angle(trans_ref)-np.angle(trans2))))))
# # plt.show()
#
# print(np.mean(np.abs(a1[(np.abs(trans_ref)+np.abs(trans1))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans1))])))]-1)))
# print(np.mean(np.abs(a2[(np.abs(trans_ref)+np.abs(trans2))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans2))])))]-1)))