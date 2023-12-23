import graph_convergence as gra_con
import numpy as np


examples=np.array(['2D_heterogeneous_3b','piece_GdM_b','Marmousi_b','SEG_EAGE_b'])
methods=np.array(['HORK','FA','KRY','RK4','2MS','RK2','RK7'])
degree=np.arange(3,31)
Ndt=np.arange(1,51)
examples_minimum_spatial_error=([2.02*1e-7,5.55*1e-7,6.62*1e-7,8.65*1e-7])
examples_minimum_time_error=([2.92*1e-7,2.65*1e-6,1.3*1e-6,4.2*1e-6])
delta=np.array([1,0.8,0.8,0.8])
T=np.array([2.2,3,3,4])

examples=np.array(['piece_GdM_b','SEG_EAGE_b'])
examples_minimum_spatial_error=([5.55*1e-7,8.65*1e-7])
examples_minimum_time_error=([2.65*1e-6,4.2*1e-6])
delta=np.array([0.8,0.8])
T=np.array([3,4])


# --------------------------------------------------------------------------------
# Velocity field graphics of Corner Model, 2D Gato do Mato, 2D Marmousi and 2D SEG/EAGE
# --------------------------------------------------------------------------------

# for i in examples:
#     gra_con.graph_velocities(i)


# --------------------------------------------------------------------------------
# Snapshots of the solution of Corner Model, 2D Gato do Mato, 2D Marmousi and 2D SEG/EAGE at time T/2
# --------------------------------------------------------------------------------

# for i in range(len(examples)):
#    gra_con.snapshot_method(example=examples[i],dx=0.005,delta=delta[i],equ='scalar_dx2',method='RK7')


# --------------------------------------------------------------------------------
# Seismogram of the solution of Corner Model, 2D Gato do Mato, 2D Marmousi and 2D SEG/EAGE at time T
# --------------------------------------------------------------------------------

# for i in range(len(examples)):
#    gra_con.seismogram(example=examples[i],delta=delta[i],T=T[i],dx=0.005)


# --------------------------------------------------------------------------------
# Minimum error of all methods related to space using the snapshot and the seismogram data
# --------------------------------------------------------------------------------
#
# for i in range(len(examples)):
#     gra_con.spatial_discr_error(example=examples[i],methods=methods,degree=degree,time_space='space',delta=delta[i])

# for i in range(len(examples)):
#     gra_con.spatial_discr_error(example=examples[i],methods=methods,degree=degree,time_space='time',delta=delta[i])


# --------------------------------------------------------------------------------
# Maximum Delta t graph for different methods using the snapshot error and seismogram
# --------------------------------------------------------------------------------
# for i in range(len(examples)):
#    gra_con.graph_methods_dt_max(example=examples[i],methods=methods,Ndt=Ndt,degree=degree,time_space='space',tol=examples_minimum_spatial_error[i]*1.5,delta=delta[i])

# gra_con.graph_methods_dt_max(example=examples[0],methods=methods,Ndt=Ndt,degree=degree,time_space='space',tol=examples_minimum_spatial_error[0]*1.5,delta=delta[0])
# gra_con.graph_methods_dt_max(example=examples[1],methods=methods,Ndt=Ndt,degree=degree[:19:2],time_space='space',tol=examples_minimum_spatial_error[1]*1.5,delta=delta[1])
# gra_con.graph_methods_dt_max(example=examples[2],methods=methods,Ndt=Ndt,degree=degree,time_space='space',tol=examples_minimum_spatial_error[2]*1.5,delta=delta[2])
# gra_con.graph_methods_dt_max(example=examples[3],methods=methods,Ndt=Ndt,degree=degree[:19:2],time_space='space',tol=examples_minimum_spatial_error[3]*1.5,delta=delta[3])

# for i in range(len(examples)):
#    gra_con.graph_methods_dt_max(example=examples[i],methods=methods,Ndt=Ndt,degree=degree,time_space='time',tol=examples_minimum_time_error[i]*1.5,delta=delta[i])

# gra_con.graph_methods_dt_max(example=examples[0],methods=methods,Ndt=Ndt,degree=degree,time_space='time',tol=examples_minimum_time_error[0]*1.5,delta=delta[0])
# gra_con.graph_methods_dt_max(example=examples[1],methods=methods,Ndt=Ndt,degree=degree[:19:2],time_space='time',tol=examples_minimum_time_error[1]*1.5,delta=delta[1])
# gra_con.graph_methods_dt_max(example=examples[2],methods=methods,Ndt=Ndt[:40],degree=degree,time_space='time',tol=examples_minimum_time_error[2]*1.5,delta=delta[2])
# gra_con.graph_methods_dt_max(example=examples[3],methods=methods,Ndt=Ndt,degree=degree[:19:2],time_space='time',tol=examples_minimum_time_error[3]*1.5,delta=delta[3])


# --------------------------------------------------------------------------------
# Efficiency graph for different methods using the snapshot error and seismogram
# --------------------------------------------------------------------------------

for i in range(len(examples)):
   gra_con.graph_methods_eff(example=examples[i],methods=methods,Ndt=Ndt,degree=degree,time_space='space',tol=examples_minimum_spatial_error[i]*1.5,delta=delta[i])

# gra_con.graph_methods_eff(example=examples[0],methods=methods,Ndt=Ndt,degree=degree,time_space='space',tol=examples_minimum_spatial_error[0]*1.5,delta=delta[0])
# gra_con.graph_methods_eff(example=examples[1],methods=methods,Ndt=Ndt,degree=degree[:19:2],time_space='space',tol=examples_minimum_spatial_error[1]*1.5,delta=delta[1])
# gra_con.graph_methods_eff(example=examples[2],methods=methods,Ndt=Ndt,degree=degree,time_space='space',tol=examples_minimum_spatial_error[2]*1.5,delta=delta[2])
# gra_con.graph_methods_eff(example=examples[3],methods=methods,Ndt=Ndt,degree=degree[:19:2],time_space='space',tol=examples_minimum_spatial_error[3]*1.5,delta=delta[3])


for i in range(len(examples)):
   gra_con.graph_methods_eff(example=examples[i],methods=methods,Ndt=Ndt,degree=degree,time_space='time',tol=examples_minimum_time_error[i]*1.5,delta=delta[i])

# gra_con.graph_methods_eff(example=examples[0],methods=methods,Ndt=Ndt,degree=degree,time_space='time',tol=examples_minimum_time_error[0]*1.5,delta=delta[0])
# gra_con.graph_methods_eff(example=examples[1],methods=methods,Ndt=Ndt,degree=degree[:19:2],time_space='time',tol=examples_minimum_time_error[1]*1.5,delta=delta[1])
# gra_con.graph_methods_eff(example=examples[2],methods=methods,Ndt=Ndt,degree=degree,time_space='time',tol=examples_minimum_time_error[2]*1.5,delta=delta[2])
# gra_con.graph_methods_eff(example=examples[3],methods=methods,Ndt=Ndt,degree=degree[:19:2],time_space='time',tol=examples_minimum_time_error[3]*1.5,delta=delta[3])


# --------------------------------------------------------------------------------
# Memory graph for different methods using the snapshot error and seismogram
# --------------------------------------------------------------------------------
for i in range(len(examples)):
   gra_con.graph_methods_mem(example=examples[i],methods=methods,Ndt=Ndt,degree=degree,T=T[i],time_space='space',tol=examples_minimum_spatial_error[i]*1.5,delta=delta[i])

# gra_con.graph_methods_mem(example=examples[0],methods=methods,Ndt=Ndt,degree=degree,T=T[0],time_space='space',tol=examples_minimum_spatial_error[0]*1.5,delta=delta[0])
# gra_con.graph_methods_mem(example=examples[1],methods=methods,Ndt=Ndt,degree=degree[:19:2],T=T[1],time_space='space',tol=examples_minimum_spatial_error[1]*1.5,delta=delta[1])
# gra_con.graph_methods_mem(example=examples[2],methods=methods,Ndt=Ndt,degree=degree,T=T[2],time_space='space',tol=examples_minimum_spatial_error[2]*1.5,delta=delta[2])
# gra_con.graph_methods_mem(example=examples[3],methods=methods,Ndt=Ndt,degree=degree[:19:2],T=T[3],time_space='space',tol=examples_minimum_spatial_error[3]*1.5,delta=delta[3])

for i in range(len(examples)):
   gra_con.graph_methods_mem(example=examples[i],methods=methods,Ndt=Ndt,degree=degree,T=T[i],time_space='time',tol=examples_minimum_time_error[i]*1.5,delta=delta[i])

# gra_con.graph_methods_mem(example=examples[0],methods=methods,Ndt=Ndt,degree=degree,T=T[0],time_space='time',tol=examples_minimum_time_error[0]*1.5,delta=delta[0])
# gra_con.graph_methods_mem(example=examples[1],methods=methods,Ndt=Ndt,degree=degree[:19:2],T=T[1],time_space='time',tol=examples_minimum_time_error[1]*1.5,delta=delta[1])
# gra_con.graph_methods_mem(example=examples[2],methods=methods,Ndt=Ndt,degree=degree,T=T[2],time_space='time',tol=examples_minimum_time_error[2]*1.5,delta=delta[2])
# gra_con.graph_methods_mem(example=examples[3],methods=methods,Ndt=Ndt,degree=degree[:19:2],T=T[3],time_space='time',tol=examples_time_spatial_error[3]*1.5,delta=delta[3])





example='2D_heterogeneous_3b'
methods=np.array(['FA','HORK','KRY','RK7','RK2','RK4','2MS'])
equ='scalar_dx2'
free_surf=1
ord='8'
Ndt=np.arange(1,51)
degree=np.arange(3,37)
dx=0.01
# gra_con.snapshot_method(example=example,dx=0.04,delta=0.8,equ='scalar_dx2',method='2MS')
# gra_con.graph_methods_dt_max(example,methods,equ,free_surf,ord,Ndt,degree,dx,tol=1e-7)
# gra_con.graph_methods_eff(example,methods,equ,free_surf,ord,Ndt,degree,dx,tol=1e-7)

# example='Marmousi_b'
# methods=np.array(['FA','HORK','KRY','RK7','RK2','2MS','RK4'])
# equ='scalar_dx2'
# free_surf=1
# ord='8'
# Ndt=np.arange(1,51)
# degree=np.arange(3,30)
# dx=0.01
# gra_con.graph_methods_dt_max(example,methods,equ,free_surf,ord,Ndt,degree,dx,tol=1e-7)
# gra_con.graph_methods_eff(example,methods,equ,free_surf,ord,Ndt,degree,dx,tol=1e-7)




# gra_con.seismogram(example='SEG_EAGE_b',dim=2,dx=0.04,delta=0.8,equ='scalar_dx2',ord='8',ind_source='H_amplified',Ndt_0=1,degree=10,method='2MS',free_surf=1,T=6)


# example='2D_homogeneous_0b'
# methods=np.array(['FA','HORK','KRY','RK7','RK2','2MS','RK4'])
# equ='scalar_dx2'
# free_surf=1
# Ndt=np.arange(1,51)
# degree=np.arange(3,30)
# delta=1
# no_PML=True
# #
# gra_con.snapshot_method(example=example,dx=0.005,delta=delta,equ='scalar_dx2',method='RK7')

# # gra_con.spatial_discr_error(example,methods,equ,free_surf,degree,delta=delta,no_PML=no_PML)
#
# gra_con.graph_methods_dt_max(example,methods,Ndt,degree,'space',tol=1.5*1.6*1e-7,delta=delta)




