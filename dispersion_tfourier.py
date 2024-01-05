import Methods as meth
import auxiliary_functions as aux_fun
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
import math
import os

def domain_dispersion(dx,c,test,dx_factor):
    # domain dimensions and amount of points
    # a=4*c*(Nr+3)/f0
    # b=14*c/f0

    if not os.path.isdir('Dispersion_S/'):
        os.mkdir("Dispersion_S")

    if (test==1 or test==10) or test==101:
        a=0.3
        b=0.2
    elif test==2:
        a=0.6
        b=0.8
    elif test==3:
        a=4+2*1
        b=4+1

    nx=int(round(a/dx))
    ny=int(round(b/dx))
    print('nx: ',nx)
    print('ny: ',ny)

    # spatial grid points
    x=np.linspace(dx,a,nx)
    y=np.linspace(b,dx,ny)
    X,Y=np.meshgrid(x,y)
    np.save('Dispersion_S/X.npy',X)
    np.save('Dispersion_S/Y.npy',Y)
    X=np.expand_dims(X.flatten('F'),1)
    Y=np.expand_dims(Y.flatten('F'),1)

    # velocity field depending on the example
    param=np.zeros((ny,nx))+c**2
    param=np.expand_dims(param.flatten('F'),1)

    # source term position
    # x0=4*c/f0
    # y0=7*c/f0

    if (test==1 or test==10) or test==101:
        x0=0.05
        y0=0.1
    elif test==2:
        x0=0.12
        y0=b/2
    elif test==3:
        x0=a/2
        y0=b-2*dx*dx_factor

    dt=dx/np.max(np.sqrt(param))/8

    return a,b,nx,ny,X,Y,param,dt,x0,y0


def source_disperion(x0,y0,X,Y,nx,ny,rad=0.02):
    f=aux_fun.source_x_2D(x0=x0,y0=y0,rad=rad,X=X,Y=Y,nx=nx,ny=ny,equ='scalar_dx2',delta=0)
    var0=np.zeros((len(f[:,0]),1))
    source_type='Dispersion_S'

    return var0,f,source_type


def solution_dispersion_cluster(method,degree,Ndt,dx,dx_factor,f0,test,delta):
    solution_dispersion(method,degree,Ndt,dx=dx,dx_factor=dx_factor,f0=f0,test=test,delta=delta,replace=0)


def solution_dispersion(method,degree,Ndt,dx=0.005,dx_factor=1,Nr=np.array([2]),fig_ind=0,f0=15,replace=1,test=1,free_surf=1,delta=0.0):
    # Function to perform a dispersion analysis based in Fourier transform, and comparing the "method" with a reference
    # solution (RK9-7 with dx/2). For different receptors is computed the solution in a time interval where only a wavelet
    # is recorder, together with its fourier transform.

    # INPUT:
    # method: (string) the method to compute de dispersion
    # degree: degree used of the polynomial for the FA,HORK, and Krylov methods
    # Nr: (integer) number of receptors with a spacing of c/f0, the approximated wavelength of Ricker's wavelet, where c
    #      is the velocity
    # dx: (float) spatial discretization grid space
    # dx_factor: this is to know the factor between the reference solution and the solutions of the methods
    # Nr: numer of receivers used in the simulations
    # fig_ind: indicator if an image of the wave propagation at the three time cuts (see below)is saved

    # OUTPUT:
    # 4 files .npy:
    #   1 - with the solution using "method" until time T=NS*c/f*1.1
    #   2 - with the solution using the reference method until time T=NS*c/f*1.1
    #   3 - with the estimated dissipation functions
    #   4 - with the estimated phase change functions

    # velocity of the homogeneous medium and central frequency of Ricker wavelet
    if test==3:
        c=3
    else:
        c=0.2
    # t0=1.2/f0+0.1
    t0=0.18
    param_ricker=np.array([f0,t0])

    # parameters of the domain where the numerical dispersion is computed
    a,b,nx,ny,X,Y,param,dt,x0,y0=domain_dispersion(dx,c,test,dx_factor)

    # parameters of the source type (Ricker wavelet)
    var0,f,source_type=source_disperion(x0,y0,X,Y,nx,ny,0.02)

    # time cuts for the three reciever positions register the wave
    if test==1:
        cuts_0=np.array([0.25,0.5,0.75])
        cuts=np.array([0.75,1,1.25])
    elif test==10:
        cuts_0=np.array([0.25,0.5,0.5])
        cuts=np.array([0.75,1,0.75])
    elif test==2:
        cuts_0=np.array([0.25,0.5,1.7])
        cuts=np.array([0.25,0.5,2.5])
    elif test==101:
        cuts_0=np.array([0.25,0.5,1.25])
        cuts=np.array([0.75,1,1.25])
    elif test==3:
        cuts_0=np.array([0,0,0.6])
        cuts=np.array([0,0,1.3])

    for i in Nr:
        # time steps given a smaller CFL condition
        T=cuts_0[i]
        dt*=Ndt
        print('dt_0: ',dt)
        NDt=np.ceil(T/dt).astype(int)
        Dt=T/NDt
        print('NDt[0]: ',NDt[0])

        # receivers positions
        if (test==1 or test==10) or test==101:
            points=np.array([np.argmin(pow(X-(i+2)*0.05,2)+pow(Y-0.1,2))])
        elif test==2:
            points=np.array([np.argmin(pow(X-(i+2)*x0,2)+pow(Y-b/2,2))])
        elif test==3:
            points=np.array([np.argmin(pow(X-a/2,2)+pow(Y-2.5,2))])

        # code names of the methods and indicators of order
        meth_ind,meth_label=method_label(method)

        # if replace==0:
        #     if meth_ind<10:
        #         if os.path.isfile('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy'):
        #             return 0
        #     else:
        #         if os.path.isfile('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy'):
        #             return 0
        #     replace=1

        # solution with a larger time step
        var=method_sol_call(method,var0,Ndt,NDt,Dt,dx,param,nx,ny,f,param_ricker,source_type,points,degree,delta,ind_minus=1,replace=replace,free_surf=free_surf)

        if cuts_0[i]<cuts[i] and f0!=15.00001:
            # time steps given a smaller CFL condition
            T=cuts[i]-cuts_0[i]
            dt/=Ndt
            print('dt_0: ',dt)
            NDt=np.ceil(T/dt).astype(int)
            Dt=T/NDt
            print('NDt[0]: ',NDt[0])
            param_ricker=np.array([f0,t0-cuts_0[i]])

            # solution with a larger time step
            method_sol_call(method,var,Ndt,NDt,Dt,dx,param,nx,ny,f,param_ricker,source_type,points,degree,delta,replace=replace,free_surf=free_surf)

            # loading again the solution for calculating the transform
            sol=method_sol_load(meth_ind,meth_label,Ndt,dx,degree,free_surf)[::dx_factor,:]
        else:
            sol=var

        if replace==1:
            if meth_ind<10:
                np.save('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(i)+'_f0_'+str(f0)+'.npy',sol)
                transform=np.fft.fft(sol[:,0])
                np.save('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(i)+'_f0_'+str(f0)+'_transform.npy',transform[:round(len(transform)/2+0.1)])
            else:
                np.save('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_points_'+str(i)+'_f0_'+str(f0)+'.npy',sol)
                transform=np.fft.fft(sol[:,0])
                np.save('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_points_'+str(i)+'_f0_'+str(f0)+'_transform.npy',transform[:round(len(transform)/2+0.1)])

            if meth_ind<10:
                sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'.npy')
                np.save('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy',sol)
            else:
                if method=='FA':
                    sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_H_amplified_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'.npy')
                else:
                    sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'.npy')
                np.save('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy',sol)
        else:
            if meth_ind<10:
                sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy')
            else:
                sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy')
        if fig_ind==1: # saving a figure of 2D wave propagation
            plt.rcParams["figure.figsize"] = [8, 5]
            ax=plt.gca()
            fig=plt.imshow(sol.reshape((ny,nx),order='F'),extent=[0,a,b,0], aspect='auto')
            plt.scatter(np.array([x0]),np.array([y0]),s=50,color='b')
            plt.scatter(np.expand_dims(X.flatten('F'),1)[points],np.expand_dims(Y.flatten('F'),1)[points],s=50,color='k',marker='s')
            plt.xlabel('X Position [km]',fontsize=24)
            plt.ylabel('Depth [km]',fontsize=24)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            cbar=plt.colorbar(fig)
            cbar.set_label('Displacement [km/s]',fontsize=24)
            cbar.ax.tick_params(labelsize=24)
            cbar.ax.yaxis.get_offset_text().set(size=16)
            plt.subplots_adjust(left=0.18, bottom=0.15, right=0.95, top=0.95)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            cbar.ax.tick_params(labelsize=15)
            plt.savefig('Dispersion_S_images/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.pdf')
            plt.show()


def solution_dispersion_wave_repr(method,degree,Ndt,dx=0.005,dx_factor=1,Nr=np.array([2]),fig_ind=0,free_surf=1):
    # Function to perform a dispersion analysis based in Fourier transform, and comparing the "method" with a reference
    # solution (RK9-7 with dx/2). For different receptors is computed the solution in a time interval where only a wavelet
    # is recorder, together with its fourier transform.

    # INPUT:
    # method: (string) the method to compute de dispersion
    # degree: degree used of the polynomial for the FA,HORK, and Krylov methods
    # Nr: (integer) number of receptors with a spacing of c/f0, the approximated wavelength of Ricker's wavelet, where c
    #      is the velocity
    # dx: (float) spatial discretization grid space
    # dx_factor: this is to know the factor between the reference solution and the solutions of the methods
    # Nr: numer of receivers used in the simulations
    # fig_ind: indicator if an image of the wave propagation at the three time cuts (see below)is saved

    # OUTPUT:
    # 4 files .npy:
    #   1 - with the solution using "method" until time T=NS*c/f*1.1
    #   2 - with the solution using the reference method until time T=NS*c/f*1.1
    #   3 - with the estimated dissipation functions
    #   4 - with the estimated phase change functions

    # velocity of the homogeneous medium and central frequency of Ricker wavelet
    c=0.2
    f0=20
    t0=1.2/f0+0.1
    param_ricker=np.array([f0,t0])

    # parameters of the domain where the numerical dispersion is computed
    a,b,nx,ny,X,Y,param,dt,x0,y0=domain_dispersion(dx,c)

    # parameters of the source type (Ricker wavelet)
    var0,f,source_type=source_disperion(x0,y0,X,Y,nx,ny,0.02)

    # time cuts for the three reciever positions register the wave
    cuts_0=np.array([0.2710084,0.52209677,0.77277487])
    cuts=np.array([0.75,1,1.25])

    for i in Nr:
        # time steps given a smaller CFL condition
        T=cuts[i]
        dt*=Ndt
        print('dt_0: ',dt)
        NDt=np.ceil(T/dt).astype(int)
        Dt=T/NDt
        print('NDt[0]: ',NDt[0])

        # receivers positions
        points=np.array([np.argmin(pow(X-(i+2)*0.05,2)+pow(Y-0.1,2))])

        # code names of the methods and indicators of order
        meth_ind,meth_label=method_label(method)

        # solution with a larger time step
        method_sol_call(method,var0,Ndt,NDt,Dt,dx,param,nx,ny,f,param_ricker,source_type,points,degree)

        # loading again the solution for calculating the transform
        sol=method_sol_load(meth_ind,meth_label,Ndt,dx,degree)[::dx_factor,:]

        if meth_ind<10:
            np.save('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(i)+'.npy',sol)
            transform=np.fft.fft(sol[:,0])
            np.save('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(i)+'_transform.npy',transform[:round(len(transform)/2+0.1)])
        else:
            np.save('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_points_'+str(i)+'.npy',sol)
            transform=np.fft.fft(sol[:,0])
            np.save('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_points_'+str(i)+'_transform.npy',transform[:round(len(transform)/2+0.1)])

        if fig_ind==1: # saving a figure of 2D wave propagation
            if meth_ind<10:
                sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'.npy')
            else:
                sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'.npy')
            plt.imshow(sol.reshape((ny,nx),order='F'),extent=[0,0.3,0,0.2], aspect='auto')
            plt.scatter(np.array([x0]),np.array([y0]),s=50,color='b')
            plt.scatter(np.expand_dims(X.flatten('F'),1)[points],np.expand_dims(Y.flatten('F'),1)[points],s=50,color='k',marker='s')
            plt.savefig('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'.pdf')
            plt.show()


def method_label(method):
    if method=='RK7':
        return 9,'RK_ref'
    elif method=='RK2':
        return 3,'sol_rk2'
    elif method=='RK4':
        return 4,'sol_rk4'
    elif method=='2MS':
        return 1,'sol_2MS'
    elif method=='FA':
        return 10,'sol_faber'
    elif method=='HORK':
        return 10,'sol_rk'
    elif method=='KRY':
        return 10,'sol_krylov'


def method_label_graph(methods):

    methods_rename=['']*len(methods)

    for i in range(len(methods)):
        if methods[i]=='RK7':
            methods_rename[i]='RK9-7'
        elif methods[i]=='RK2':
            methods_rename[i] = 'RK3-2'
        elif methods[i]=='RK4':
            methods_rename[i] = 'RK4-4'
        elif methods[i]=='2MS':
            methods_rename[i] = 'Leap-frog'
        elif methods[i]=='KRY':
            methods_rename[i]=r"KRY$^*$"
        else:
            methods_rename[i] = methods[i]
    return methods_rename


def method_sol_call(method,var0,Ndt,NDt,Dt,dx,param,nx,ny,f,param_ricker,source_type,points,degree,delta,ind_minus=0,replace=1,free_surf=1):
    example='Dispersion_S'
    equ='scalar_dx2'
    T_frac_snapshot=1  # this value is only to not save the velocity field
    ord='8'
    dim=2
    beta0=30
    ind_source='H_amplified'

    var_return=meth.method_solver(method,var0,Ndt,NDt,Dt,T_frac_snapshot,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example,degree,ind_source,replace,False)
    if method=="2MS" and ind_minus==1:
        var_minus1=meth.method_solver(method,var0,Ndt,NDt-1,Dt,T_frac_snapshot,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example,degree,ind_source,replace,False)
        np.save('Dispersion_S/2MS_minus1.npy',var_minus1)
    return var_return


def method_sol_load(meth_ind,meth_label,Ndt,dx,degree,free_surf):
    if meth_ind<10:
        return np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points.npy')
    else:
        if meth_label=='sol_faber':
            return np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_H_amplified_points_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'.npy')
        else:
            return np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_points_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'.npy')


def graph_wave_disp_diss(method,degree,Ndt,dx=0.005,Nr=np.array([2]),f0=15,dt=0.003125,results='dispersion',free_surf=1,dx_factor=4):
    # Function to compare an approximate solution with the reference solution

    meth_ind,meth_label=method_label(method)
    cuts_0=np.array([0.25,0.5,0.75])
    cuts=np.array([0.75,1,1.25])

    if 'convergence' in results:
        for i in range(len(Nr)):
            sol_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_1_dx_'+str(dx/dx_factor)+'_points_'+str(Nr[i])+'_f0_'+str(f0)+'.npy')
            if meth_ind<10:
                sol=np.load('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(Nr[i])+'_f0_'+str(f0)+'.npy')
            else:
                sol=np.load('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_points_'+str(Nr[i])+'_f0_'+str(f0)+'.npy')
            plt.plot(np.linspace(cuts_0[i],cuts_0[i]+cuts[i],len(sol_ref)),sol_ref,label='Reference',linewidth=2)
            plt.plot(np.linspace(cuts_0[i],cuts_0[i]+cuts[i],len(sol)),sol,label=method,linewidth=2)
            plt.legend()
            plt.ylabel('Displacement',fontsize=16)
            plt.xlabel('Time',fontsize=16)
            plt.savefig('Dispersion_S_images/'+method+'_dx_'+str(dx)+'_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_points_'+str(i)+'_f0_'+str(f0)+'.pdf')
            plt.show()
            # plt.clf()

    if 'dissipation' in results:
        for i in range(len(Nr)):
            trans_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_1_dx_'+str(dx/dx_factor)+'_points_'+str(Nr[i])+'_f0_'+str(f0)+'_transform.npy')
            if meth_ind<10:
                trans=np.load('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(Nr[i])+'_f0_'+str(f0)+'_transform.npy')
            else:
                trans=np.load('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_points_'+str(Nr[i])+'_f0_'+str(f0)+'_transform.npy')
            # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
            # a=trans_ref/trans
            # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=1
            # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
            # print('dissipation',np.mean(np.abs(a[(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]-1)))
            # plt.plot(np.arange(len(trans))/(cuts[i]-cuts_0[i]),np.abs(a),label='Receiver_'+str(Nr[i]),linewidth=2)
            if len(trans)>len(trans_ref):
                trans=trans[:-1]
            ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
            if 'dissipation1' in results:
                print('dispersion',np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq])))*(cuts[i]-cuts_0[i])/len(trans))
                plt.plot(np.arange(len(trans))[ind_freq]/(cuts[i]-cuts_0[i]),np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq])),label='Receiver_'+str(Nr[i]),linewidth=2)
            elif 'dissipation2' in results:
                print('dispersion',np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq]))*np.abs(np.log(np.abs(trans_ref[ind_freq]))-np.log(np.abs(trans[ind_freq]))))*(cuts[i]-cuts_0[i])/len(trans))
                plt.plot(np.arange(len(trans))[ind_freq]/(cuts[i]-cuts_0[i]),np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq]))*np.abs(np.log(np.abs(trans_ref[ind_freq]))-np.log(np.abs(trans[ind_freq]))),label='Receiver_'+str(Nr[i]),linewidth=2)
        plt.legend()
        plt.ylabel('Dissipation',fontsize=16)
        plt.xlabel('Frequency',fontsize=16)
        plt.savefig('Dispersion_S_images/'+method+'_dx_'+str(dx)+'_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_f0_'+str(f0)+'_amplitude.pdf')
        plt.show()
        # plt.clf()

    if 'dispersion' in results:
        for i in range(len(Nr)):
            trans_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_1_dx_'+str(dx/dx_factor)+'_points_'+str(Nr[i])+'_f0_'+str(f0)+'_transform.npy')
            if meth_ind<10:
                trans=np.load('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(Nr[i])+'_f0_'+str(f0)+'_transform.npy')
            else:
                trans=np.load('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_points_'+str(Nr[i])+'_f0_'+str(f0)+'_transform.npy')
            if len(trans)>len(trans_ref):
                trans=trans[:-1]
            # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
            # a=trans_ref/trans
            # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=1.0
            # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
            # print('dispersion',np.mean(np.abs(np.angle(a))))
            # plt.plot(np.arange(len(trans))/(cuts[i]-cuts_0[i]),np.angle(a),label='Receiver_'+str(Nr[i]),linewidth=2)
            ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
            print('dispersion',np.sum(np.abs(np.angle(np.exp(1j*(np.angle(trans_ref[ind_freq])-np.angle(trans[ind_freq]))))))*(cuts[i]-cuts_0[i])/len(trans))
            plt.plot(np.arange(len(trans))[ind_freq]/(cuts[i]-cuts_0[i]),np.angle(np.exp(1j*(np.angle(trans_ref[ind_freq])-np.angle(trans[ind_freq])))),label='Receiver_'+str(Nr[i]),linewidth=2)
        plt.legend()
        plt.ylabel('Dispersion',fontsize=16)
        plt.xlabel('Frequency',fontsize=16)
        plt.savefig('Dispersion_S_images/'+method+'_dx_'+str(dx)+'_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_f0_'+str(f0)+'_phase.pdf')
        plt.show()
        # plt.clf()


def graph_estimate_diss_disp(methods,degree,Ndt,dx=0.005,Nr_ind=2,fig_ind='methods',f0=15,results='dispersion',free_surf=1,test=1,dx_factor=4):
    # Function to calculate the spatial error

    trans_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_1_dx_'+str(dx/dx_factor)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
    sol_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_1_dx_'+str(dx/dx_factor)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'.npy')
    # sol_ref = sol_ref[::2, :]
    # np.save('Dispersion_S/RK7_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx/dx_factor)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'.npy',sol_ref)
    # transform=np.fft.fft(sol_ref[:,0])
    # np.save('Dispersion_S/RK7_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx/dx_factor)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy',transform[:round(len(transform)/2+0.1)])
    # gdfhd

    cuts_0=np.array([0.25,0.5,0.75])
    cuts=np.array([0.75,1,1.25])

    color=np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf'])
    names_graph=method_label_graph(methods)
    marker=np.array(['D','o','s','v','P','H','D'])
    line=np.array(['-','--','-.'])
    line_width=3
    marker_size = 40

    plt.rcParams["figure.figsize"] = [6.5,5]

    if 'convergence1' in results:
        count_high_order=0
        ax=plt.gca()
        for i in range(len(methods)):
            meth_ind,meth_label=method_label(methods[i])
            if meth_ind<10:
                sol=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'.npy')
                error=np.max(np.abs(sol-sol_ref))
                ax.scatter(meth_ind,error, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],s=marker_size*1.7,zorder=5*i)
                ax.scatter(meth_ind,error,marker=marker[i],color=color[i],s=marker_size,zorder=5*i)
                ax.scatter([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],zorder=5*i)
            else:
                error=np.zeros(len(degree))
                for j in range(len(degree)):
                    sol=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'.npy')
                    error[j]=np.max(np.abs(sol-sol_ref))
                lin,=ax.plot(degree,error,linewidth=line_width,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
                ax.scatter(degree,error, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
                ax.plot([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
                count_high_order+=1
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.legend(fontsize=22)
        plt.ylabel(r'$Err_{conv}$',fontsize=22)
        plt.xlabel('# Stages',fontsize=20)
        ax.yaxis.get_offset_text().set(size=17)
        plt.subplots_adjust(left=0.18, bottom=0.15, right=0.95, top=0.95)
        plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'_error.pdf')
        plt.show()
        plt.clf()
    if 'convergence2' in results:
        sol_ref_c2=np.load('Dispersion_S/RK_ref_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_1_dx_'+str(dx/dx_factor)+'_f0_'+str(f0)+'.npy')
        a,b,nx,ny,X,Y,param,dt,x0,y0=domain_dispersion(dx/dx_factor,0.2,test=test)
        sol_ref_c2=sol_ref_c2.reshape((ny,nx),order='F')
        sol_ref_c2=sol_ref_c2[1::dx_factor,:-1:dx_factor]
        sol_ref_c2=sol_ref_c2.flatten('F')
        count_high_order=0
        ax=plt.gca()
        for i in range(len(methods)):
            meth_ind,meth_label=method_label(methods[i])
            if meth_ind<10:
                sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy')
                error=np.max(np.abs(sol-sol_ref_c2))
                ax.scatter(meth_ind,error, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],s=marker_size*1.7,zorder=5*i)
                ax.scatter(meth_ind,error,marker=marker[i],color=color[i],s=marker_size,zorder=5*i)
                ax.scatter([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],zorder=5*i)
            else:
                error=np.zeros(len(degree))
                for j in range(len(degree)):
                    sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy')
                    error[j]=np.max(np.abs(sol-sol_ref_c2))
                lin,=ax.plot(degree,error,linewidth=line_width,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
                ax.scatter(degree,error, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
                ax.plot([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
                count_high_order+=1
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.ylabel(r'$Err_{max}$',fontsize=22)
        plt.xlabel('# Stages',fontsize=20)
        ax.yaxis.get_offset_text().set(size=17)
        plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)
        plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'_error2.pdf')
        plt.show()
        plt.clf()


    if 'dispersion' in results:
        count_high_order=0
        ax=plt.gca()
        for i in range(len(methods)):
            meth_ind,meth_label=method_label(methods[i])
            if meth_ind<10:
                trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
                if len(trans)>len(trans_ref):
                    trans=trans[:-1]
                # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
                # a=trans_ref/trans
                # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=1.0
                # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
                # dispersion_mes=np.mean(np.abs(np.angle(a[(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))])))
                ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
                dispersion_mes=np.sum(np.abs(np.angle(np.exp(1j*(np.angle(trans_ref[ind_freq])-np.angle(trans[ind_freq]))))))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                ax.scatter(meth_ind,dispersion_mes, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],s=marker_size*1.7,zorder=5*i)
                ax.scatter(meth_ind,dispersion_mes,marker=marker[i],color=color[i],s=marker_size,zorder=5*i)
                ax.scatter([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],zorder=5*i)
            else:
                dispersion_mes=np.zeros(len(degree))
                for j in range(len(degree)):
                    trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
                    if len(trans)>len(trans_ref):
                        trans=trans[:-1]
                    # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
                    # a=trans_ref/trans
                    # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=1.0
                    # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
                    # dispersion_mes[j]=np.mean(np.abs(np.angle(a[(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))])))
                    ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
                    dispersion_mes[j]=np.sum(np.abs(np.angle(np.exp(1j*(np.angle(trans_ref[ind_freq])-np.angle(trans[ind_freq]))))))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                lin,=ax.plot(degree,dispersion_mes,linewidth=line_width,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
                ax.scatter(degree,dispersion_mes, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
                ax.plot([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
                count_high_order+=1
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.legend(fontsize=20,loc='upper right')
        plt.ylabel(r'$Err_{disp}$',fontsize=22)
        plt.xlabel('# Stages (MVOs)',fontsize=20)
        ax.yaxis.get_offset_text().set(size=17)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)
        plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'_dispersion.pdf')
        plt.show()
        # plt.clf()

    plt.rcParams["figure.figsize"] = [10.5, 5]
    plt.show()
    if 'dissipation' in results:
        count_high_order=0
        ax=plt.gca()
        for i in range(len(methods)):
            meth_ind,meth_label=method_label(methods[i])
            if meth_ind<10:
                trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
                if len(trans)>len(trans_ref):
                    trans=trans[:-1]
                # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
                # a=trans_ref/trans
                # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=0
                # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
                # dissipation_mes=np.mean(np.abs(a-1))
                ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
                if 'dissipation1' in results:
                    dissipation_mes=np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq])))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                elif 'dissipation2' in results:
                    dissipation_mes=np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq]))*np.abs(np.log(np.abs(trans_ref[ind_freq]))-np.log(np.abs(trans[ind_freq]))))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                ax.scatter(meth_ind,dissipation_mes, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],s=marker_size*1.7,zorder=5*i)
                ax.scatter(meth_ind,dissipation_mes,marker=marker[i],color=color[i],s=marker_size,zorder=5*i)
                ax.scatter([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],zorder=5*i)
            else:
                dissipation_mes=np.zeros(len(degree))
                for j in range(len(degree)):
                    trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
                    if len(trans)>len(trans_ref):
                        trans=trans[:-1]
                    # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
                    # a=trans_ref/trans
                    # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=0
                    # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
                    # dissipation_mes[j]=np.mean(np.abs(a-1))
                    ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
                    if 'dissipation1' in results:
                        dissipation_mes[j]=np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq])))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                    elif 'dissipation2' in results:
                        dissipation_mes[j]=np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq]))*np.abs(np.log(np.abs(trans_ref[ind_freq]))-np.log(np.abs(trans[ind_freq]))))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                lin,=ax.plot(degree,dissipation_mes,linewidth=line_width,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
                ax.scatter(degree,dissipation_mes, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
                ax.plot([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
                count_high_order+=1
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.yaxis.get_offset_text().set(size=17)
        plt.legend(fontsize=20,loc='center left', bbox_to_anchor=(1.2, 0.5))
        plt.ylabel(r'$Err_{diss}$',fontsize=22)
        plt.xlabel('# Stages (MVOs)',fontsize=20)
        plt.subplots_adjust(left=0.115, bottom=0.15, right=0.6, top=0.95)
        plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'_dissipation.pdf')
        plt.show()


def tolerances(f0,cut='new'):
    # convergence1,convergence2,dispersion,dissipation1,dissipation2
    if cut=="new":
        if f0==10:
            return np.array([7.125*1e-7,0.001,3.78*1e-7,0])
        elif f0==15:
            return np.array([3.9*1e-6,0.002,2.4*1e-7,0])   # only this oone in this conditional is correct 1.86*1e-5 convergence2
        elif f0==20:
            return np.array([3.5*1e-6,0.042,2.4*1e-7,0])
        elif f0==25:
            return np.array([4.1*1e-6,0.075,2.2*1e-7,0])
        elif f0==30:
            return np.array([0,0,0,0])
    elif cut=="max":
        if f0==10:
            return np.array([2*1e-5,0.0626,7*1e-7,2.15*1e-7])
        elif f0==15:
            return np.array([8*1e-6,0.1418,8.125*1e-7,8.37*1e-7])
        elif f0==20:
            return np.array([4.2*1e-6,0.1716,6.87*1e-7,9.95*1e-7])
        elif f0==25:
            return np.array([2.47*1e-6,0.2467,5.22*1e-7,8.81*1e-7])
        elif f0==30:
            return np.array([1.6*1e-6,0.2619,3.84*1e-7,7.12*1e-7])
    else:
        if f0==10:
            return np.array([6*1e-6,0.0626,7*1e-7,2.15*1e-7])
        elif f0==15:
            return np.array([3.1*1e-6,0.1418,8.125*1e-7,8.37*1e-7])
        elif f0==20:
            return np.array([2.05*1e-6,0.1716,6.87*1e-7,9.95*1e-7])
        elif f0==25:
            return np.array([1.29*1e-6,0.2467,5.22*1e-7,8.81*1e-7])
        elif f0==30:
            return np.array([8.66*1e-7,0.2619,3.84*1e-7,7.12*1e-7])


def graph_estimate_diss_disp_max_dt(methods,degree,Ndt,dx=0.005,Nr_ind=2,fig_ind='methods',dt=0.003125,f0=15,results='dispersion',free_surf=1,tol_mult=1.5,dx_factor=4,test=1):
    tol_err,tol_disp,tol_diss1,tol_diss2=tolerances(f0)*tol_mult
    # tol_err2=2.2*1e-5*tol_mult
    # tol_err2=2.35*1e-5*tol_mult
    # tol_err2=4*1e-5*tol_mult
    tol_err2=6.8e-7*tol_mult

    trans_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx/dx_factor)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
    sol_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_1_dx_'+str(dx/dx_factor)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'.npy')

    cuts_0=np.array([0.25,0.5,0.75])
    cuts=np.array([0.75,1,1.25])

    color=np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf'])
    names_graph=method_label_graph(methods)
    line_width=3
    marker_size=40
    marker=np.array(['D','o','s','v','P','H','D'])
    line=np.array(['-','--','-.'])
    plt.rcParams["figure.figsize"] = [10,5]

    if 'convergence1' in results:
        count_high_order=0
        ax=plt.gca()
        for i in range(len(methods)):
            meth_ind,meth_label=method_label(methods[i])
            if meth_ind<10:
                max_dt=0
                for j in range(len(Ndt)):
                    sol=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'.npy')
                    if len(sol)>len(sol_ref):
                        sol=sol[:-1]
                    error=np.max(np.abs(sol-sol_ref))
                    if error>tol_err  or math.isnan(error):
                        break
                    else:
                        max_dt=dt*Ndt[j]
                ax.scatter(meth_ind,max_dt, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],s=marker_size*1.7,zorder=5*i)
                ax.scatter(meth_ind,max_dt,marker=marker[i],color=color[i],s=marker_size,zorder=5*i)
                ax.scatter([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],zorder=5*i)
            else:
                max_dt=np.zeros(len(degree))
                for j in range(len(degree)):
                    for k in range(len(Ndt)):
                        try:
                            sol=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[k])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'.npy')
                        except:
                            break
                        if len(sol)>len(sol_ref):
                            sol=sol[:-1]
                        error=np.max(np.abs(sol-sol_ref))
                        if error>tol_err  or math.isnan(error):
                            break
                        else:
                            max_dt[j]=dt*Ndt[k]
                lin,=ax.plot(degree,max_dt,linewidth=line_width,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
                ax.scatter(degree,max_dt, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
                ax.plot([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
                count_high_order+=1
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(fontsize=21,loc='center left', bbox_to_anchor=(1, 0.5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel(r'$\Delta t_{max}(E_{rr})$',fontsize=24)
        plt.xlabel('# Stages (MVOs)',fontsize=22)
        plt.subplots_adjust(left=0.2, bottom=0.15, right=0.7, top=0.95)
        plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_dx_'+str(dx)+'_f0_'+str(f0)+'_error_max_dt.pdf')
        plt.show()
        plt.clf()
    if 'convergence2' in results:
        sol_ref=np.load('Dispersion_S/RK_ref_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_1_dx_'+str(dx/dx_factor)+'_f0_'+str(f0)+'.npy')
        a,b,nx,ny,X,Y,param,dt,x0,y0=domain_dispersion(dx/dx_factor,0.2,test=test)
        sol_ref=sol_ref.reshape((ny,nx),order='F')
        sol_ref=sol_ref[1::dx_factor,:-1:dx_factor]
        sol_ref=sol_ref.flatten('F')
        count_high_order=0
        ax=plt.gca()
        for i in range(len(methods)):
            meth_ind,meth_label=method_label(methods[i])
            if meth_ind<10:
                max_dt=0
                for j in range(len(Ndt)):
                    sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[j])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy')
                    error=np.max(np.abs(sol-sol_ref))
                    if error>tol_err2 or math.isnan(error):
                        break
                    else:
                        max_dt=dt*Ndt[j]
                ax.scatter(meth_ind,max_dt, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],s=marker_size*1.7,zorder=5*i)
                ax.scatter(meth_ind,max_dt,marker=marker[i],color=color[i],s=marker_size,zorder=5*i)
                ax.scatter([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],zorder=5*i)
            else:
                max_dt=np.zeros(len(degree))
                for j in range(len(degree)):
                    for k in range(len(Ndt)):
                        try:
                            sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[k])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy')
                        except:
                            break
                        if len(sol)>len(sol_ref):
                            sol=sol[:-1]
                        error=np.max(np.abs(sol-sol_ref))
                        if error>tol_err2 or math.isnan(error):
                            break
                        else:
                            max_dt[j]=dt*Ndt[k]
                lin,=ax.plot(degree,max_dt,linewidth=line_width,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
                ax.scatter(degree,max_dt, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
                ax.plot([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
                count_high_order+=1
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.ylabel(r'$\Delta t_{max}(E_{rr})$',fontsize=22)
        plt.xlabel('# Stages (MVOs)',fontsize=20)
        plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.95)
        plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_dx_'+str(dx)+'_f0_'+str(f0)+'_error2_max_dt.pdf')
        plt.show()
        # plt.clf()

    # plt.rcParams["figure.figsize"] = [10,8]     #this is only for the legend figure
    plt.rcParams["figure.figsize"] = [7,5]
    plt.show()
    if 'dispersion' in results:
        count_high_order=0
        ax=plt.gca()
        for i in range(len(methods)):
            meth_ind,meth_label=method_label(methods[i])
            if meth_ind<10:
                max_dt=0
                for j in range(len(Ndt)):
                    trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
                    if len(trans)>len(trans_ref):
                        trans=trans[:-1]
                    # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
                    # a=trans_ref/trans
                    # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=1.0
                    # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
                    # dispersion_mes=np.mean(np.abs(np.angle(a)))
                    ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
                    dispersion_mes=np.sum(np.abs(np.angle(np.exp(1j*(np.angle(trans_ref[ind_freq])-np.angle(trans[ind_freq]))))))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                    if dispersion_mes>tol_disp or math.isnan(dispersion_mes):
                        break
                    else:
                        max_dt=dt*Ndt[j]
                ax.scatter(meth_ind,max_dt, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],s=marker_size*1.7,zorder=5*i)
                ax.scatter(meth_ind,max_dt,marker=marker[i],color=color[i],s=marker_size,zorder=5*i)
                ax.scatter([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],zorder=5*i)
                # plt.scatter(np.array([meth_ind]),np.array([max_dt]),label=methods[i],marker='s',s=40)
            else:
                max_dt=np.zeros(len(degree))
                for j in range(len(degree)):
                    for k in range(len(Ndt)):
                        try:
                            trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[k])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
                        except:
                            break
                        if len(trans)>len(trans_ref):
                            trans=trans[:-1]
                        elif len(trans_ref)>len(trans):
                            trans_ref=trans_ref[:-1]
                        # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
                        # a=trans_ref/trans
                        # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=1.0
                        # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
                        # dispersion_mes=np.mean(np.abs(np.angle(a)))
                        ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
                        dispersion_mes=np.sum(np.abs(np.angle(np.exp(1j*(np.angle(trans_ref[ind_freq])-np.angle(trans[ind_freq]))))))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                        if dispersion_mes>tol_disp or math.isnan(dispersion_mes):
                            break
                        else:
                            max_dt[j]=dt*Ndt[k]
                lin,=ax.plot(degree,max_dt,linewidth=line_width,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
                ax.scatter(degree,max_dt, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
                ax.plot([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
                count_high_order+=1
                # plt.plot(degree,max_dt,label=methods[i],linewidth=2)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.legend(fontsize=20,loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4, fancybox=True, shadow=True)  #this is only for the legend figure
        plt.ylabel(r'$\Delta t_{max}$',fontsize=22)
        plt.xlabel('# Stages (MVOs)',fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)
        # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8) #this is only for the legend figure
        plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_dx_'+str(dx)+'_f0_'+str(f0)+'_dispersion_max_dt.pdf')
        # plt.savefig('legend_b.pdf')   #this is only for the legend figure
        plt.show()
        # plt.clf()

    if 'dissipation' in results:
        count_high_order=0
        ax=plt.gca()
        for i in range(len(methods)):
            meth_ind,meth_label=method_label(methods[i])
            if meth_ind<10:
                max_dt=0
                for j in range(len(Ndt)):
                    trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
                    if len(trans)>len(trans_ref):
                            trans=trans[:-1]
                    # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
                    # a=trans_ref/trans
                    # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=0
                    # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
                    # dissipation_mes=np.mean(np.abs(a-1))
                    ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
                    if 'dissipation1' in results:
                        dissipation_mes=np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq])))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                        if dissipation_mes>tol_diss1 or math.isnan(dissipation_mes):
                            break
                        else:
                            max_dt=dt*Ndt[j]
                    elif 'dissipation2' in results:
                        dissipation_mes=np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq]))*np.abs(np.log(np.abs(trans_ref[ind_freq]))-np.log(np.abs(trans[ind_freq]))))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                        if dissipation_mes>tol_diss2:
                            break
                        else:
                            max_dt=dt*Ndt[j]
                print(meth_ind,dt*Ndt[j],max_dt)
                ax.scatter(meth_ind,max_dt, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],s=marker_size*1.7,zorder=5*i)
                ax.scatter(meth_ind,max_dt,marker=marker[i],color=color[i],s=marker_size,zorder=5*i)
                ax.scatter([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],zorder=5*i)
                # plt.scatter(np.array([meth_ind]),np.array([max_dt]),label=methods[i],marker='s',s=40)
            else:
                max_dt=np.zeros(len(degree))
                for j in range(len(degree)):
                    for k in range(len(Ndt)):
                        try:
                            trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[k])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
                        except:
                            break
                        if len(trans)>len(trans_ref):
                            trans=trans[:-1]
                        # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
                        # a=trans_ref/trans
                        # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=0
                        # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
                        # dissipation_mes=np.mean(np.abs(a))
                        ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
                        if 'dissipation1' in results:
                            dissipation_mes=np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq])))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                            if dissipation_mes>tol_diss1 or math.isnan(dissipation_mes):
                                break
                            else:
                                max_dt[j]=dt*Ndt[k]
                        elif 'dissipation2' in results:
                            dissipation_mes=np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq]))*np.abs(np.log(np.abs(trans_ref[ind_freq]))-np.log(np.abs(trans[ind_freq]))))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                            if dissipation_mes>tol_diss2:
                                break
                            else:
                                max_dt[j]=dt*Ndt[k]
                lin,=ax.plot(degree,max_dt,linewidth=line_width,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
                ax.scatter(degree,max_dt, linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
                ax.plot([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
                count_high_order+=1
                # plt.plot(degree,max_dt,label=methods[i],linewidth=2)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.legend(fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel(r'$\Delta t_{max}$',fontsize=22)
        plt.xlabel('# Stages (MVOs)',fontsize=20)
        plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)
        if 'dissipation1' in results:
            plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_dx_'+str(dx)+'_f0_'+str(f0)+'_dissipation1_max_dt.pdf')
        elif 'dissipation2' in results:
            plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_dx_'+str(dx)+'_f0_'+str(f0)+'_dissipation2_max_dt.pdf')
        plt.show()


def graph_estimate_diss_disp_eff(methods,degree,Ndt,dx=0.005,Nr_ind=2,fig_ind='methods',dt=0.003125,f0=15,results='dispersion',graph_limit=2000,free_surf=1,tol_mult=1.5):
    tol_err,tol_disp,tol_diss1,tol_diss2=tolerances(f0)*tol_mult
    trans_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx/4)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
    sol_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_1_dx_'+str(dx/4)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'.npy')

    cuts_0=np.array([0.25,0.5,0.75])
    cuts=np.array([0.75,1,1.25])

    color=np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf'])
    names_graph=method_label_graph(methods)
    marker=np.array(['D','o','s','v','P','H','D'])
    line=np.array(['-','--','-.'])
    line_width=3
    marker_size=40
    plt.rcParams["figure.figsize"] = [9,7]

    if 'convergence' in results:
        count_high_order=0
        ax=plt.gca()
        for i in range(len(methods)):
            meth_ind,meth_label=method_label(methods[i])
            if meth_ind<10:
                eff=1e5
                for j in range(len(Ndt)):
                    sol=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'.npy')
                    error=np.max(np.abs(sol-sol_ref))
                    if error>tol_err or math.isnan(error):
                        break
                    else:
                        eff=meth_ind/(dt*Ndt[j])
                ax.scatter(meth_ind,np.log10(eff), linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],s=marker_size*1.7,zorder=5*i)
                ax.scatter(meth_ind,np.log10(eff),marker=marker[i],color=color[i],s=marker_size,zorder=5*i)
                ax.scatter([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],zorder=5*i)
            else:
                eff=np.zeros(len(degree))+1e5
                for j in range(len(degree)):
                    for k in range(len(Ndt)):
                        sol=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[k])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'.npy')
                        if len(sol)>len(sol_ref):
                            sol=sol[:-1]
                        error=np.max(np.abs(sol-sol_ref))
                        if error>tol_err or math.isnan(error):
                            break
                        else:
                            eff[j]=degree[j]/(dt*Ndt[k])
                lin,=ax.plot(degree,np.log10(eff),linewidth=line_width,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
                ax.scatter(degree,np.log10(eff), linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
                ax.plot([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
                count_high_order+=1
        y_ticks_position=np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],7)
        y_ticks_labels=np.char.add([r"%.1f$\cdot$" % pow(10,math.modf(x)[0]) for x in y_ticks_position],[r"$10^{%.0f}$" % int(math.modf(x)[1]) for x in y_ticks_position])
        ax.set_yticks(y_ticks_position)
        ax.set_yticklabels(y_ticks_labels)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(top=np.minimum(graph_limit,ax.get_ylim()[1]),bottom=np.maximum(0,ax.get_ylim()[0]))
        plt.legend(fontsize=20)
        plt.ylabel(r'$N^\alpha_{op}(Err_{max})$',fontsize=22)
        plt.xlabel('# Stages',fontsize=20)
        plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.95)
        plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_dx_'+str(dx)+'_f0_'+str(f0)+'_error_eff.pdf')
        plt.show()
        plt.clf()

    plt.rcParams["figure.figsize"] = [7,5]
    plt.show()
    if 'dispersion' in results:
        count_high_order=0
        ax=plt.gca()
        for i in range(len(methods)):
            meth_ind,meth_label=method_label(methods[i])
            if meth_ind<10:
                eff=1e5
                for j in range(len(Ndt)):
                    trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')

                    if len(trans)>len(trans_ref):
                        trans=trans[:-1]
                    # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
                    # a=trans_ref/trans
                    # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=1.0
                    # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
                    # dispersion_mes=np.mean(np.abs(np.angle(a)))
                    ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
                    dispersion_mes=np.sum(np.abs(np.angle(np.exp(1j*(np.angle(trans_ref[ind_freq])-np.angle(trans[ind_freq]))))))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                    if dispersion_mes>tol_disp or math.isnan(dispersion_mes):
                        break
                    else:
                        eff=meth_ind/(dt*Ndt[j])
                ax.scatter(meth_ind,np.log10(eff), linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],s=marker_size*1.7,zorder=5*i)
                ax.scatter(meth_ind,np.log10(eff),marker=marker[i],color=color[i],s=marker_size,zorder=5*i)
                ax.scatter([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],zorder=5*i)
                # plt.scatter(np.array([meth_ind]),np.array([max_dt]),label=methods[i],marker='s',s=40)
            else:
                eff=np.zeros(len(degree))+1e5
                for j in range(len(degree)):
                    for k in range(len(Ndt)):
                        try:
                            trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[k])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
                        except:
                            break
                        if len(trans)>len(trans_ref):
                            trans=trans[:-1]
                        # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
                        # a=trans_ref/trans
                        # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=1.0
                        # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
                        # dispersion_mes=np.mean(np.abs(np.angle(a)))
                        ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
                        dispersion_mes=np.sum(np.abs(np.angle(np.exp(1j*(np.angle(trans_ref[ind_freq])-np.angle(trans[ind_freq]))))))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                        if dispersion_mes>tol_disp or math.isnan(dispersion_mes):
                            break
                        else:
                            eff[j]=degree[j]/(dt*Ndt[k])
                eff[eff>graph_limit]=float("nan")  # to remove points of the graph larger than the limit
                lin,=ax.plot(degree,np.log10(eff),linewidth=line_width,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
                ax.scatter(degree,np.log10(eff), linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
                ax.plot([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
                count_high_order+=1
                # plt.plot(degree,max_dt,label=methods[i],linewidth=2)
        y_ticks_position=np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],7)
        y_ticks_labels=np.char.add([r"%.1f$\cdot$" % pow(10,math.modf(x)[0]) for x in y_ticks_position],[r"$10^{%.0f}$" % int(math.modf(x)[1]) for x in y_ticks_position])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yticks(y_ticks_position)
        ax.set_yticklabels(y_ticks_labels)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(top=np.minimum(graph_limit,ax.get_ylim()[1]),bottom=np.maximum(0,ax.get_ylim()[0]))
        # plt.legend(fontsize=20)
        plt.ylabel(r'$N^{disp}_{op}$',fontsize=22)
        plt.xlabel('# Stages (MVOs)',fontsize=20)
        plt.subplots_adjust(left=0.24, bottom=0.15, right=0.95, top=0.95)
        plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_dx_'+str(dx)+'_f0_'+str(f0)+'_dispersion_eff.pdf')
        plt.show()
        # plt.clf()

    if 'dissipation' in results:
        count_high_order=0
        ax=plt.gca()
        for i in range(len(methods)):
            meth_ind,meth_label=method_label(methods[i])
            if meth_ind<10:
                eff=1e5
                for j in range(len(Ndt)):
                    trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
                    if len(trans)>len(trans_ref):
                        trans=trans[:-1]
                    # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
                    # a=trans_ref/trans
                    # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=0
                    # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
                    # dissipation_mes=np.mean(np.abs(a-1))
                    ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
                    if 'dissipation1' in results:
                        dissipation_mes=np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq])))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                        if dissipation_mes>tol_diss1:
                            break
                        else:
                            eff=meth_ind/(dt*Ndt[j])
                    elif 'dissipation2' in results:
                        dissipation_mes=np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq]))*np.abs(np.log(np.abs(trans_ref[ind_freq]))-np.log(np.abs(trans[ind_freq]))))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                        if dissipation_mes>tol_diss2 or math.isnan(dissipation_mes):
                            break
                        else:
                            eff=meth_ind/(dt*Ndt[j])
                print(meth_ind,dt*Ndt[j],eff)
                ax.scatter(meth_ind,np.log10(eff), linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],s=marker_size*1.7,zorder=5*i)
                ax.scatter(meth_ind,np.log10(eff),marker=marker[i],color=color[i],s=marker_size,zorder=5*i)
                ax.scatter([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],zorder=5*i)
                # plt.scatter(np.array([meth_ind]),np.array([max_dt]),label=methods[i],marker='s',s=40)
            else:
                eff=np.zeros(len(degree))+1e5
                for j in range(len(degree)):
                    for k in range(len(Ndt)):
                        try:
                            trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[k])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_f0_'+str(f0)+'_transform.npy')
                        except:
                            break
                        if len(trans)>len(trans_ref):
                            trans=trans[:-1]
                        # trans=np.hstack((trans,np.zeros(len(trans_ref)-len(trans))))
                        # a=trans_ref/trans
                        # a[(np.abs(trans_ref)+np.abs(trans))<0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=0
                        # a[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]=trans_ref[((np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))]))))*(trans==0)]
                        # dissipation_mes=np.mean(np.abs(a))
                        ind_freq=(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
                        if 'dissipation1' in results:
                            dissipation_mes=np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq])))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                            if dissipation_mes>tol_diss1 or math.isnan(dissipation_mes):
                                break
                            else:
                                eff[j]=degree[j]/(dt*Ndt[k])
                        elif 'dissipation2' in results:
                            dissipation_mes=np.sum(np.abs(np.abs(trans_ref[ind_freq])-np.abs(trans[ind_freq]))*np.abs(np.log(np.abs(trans_ref[ind_freq]))-np.log(np.abs(trans[ind_freq]))))*(cuts[Nr_ind]-cuts_0[Nr_ind])/len(trans)
                            if dissipation_mes>tol_diss2:
                                break
                            else:
                                eff[j]=degree[j]/(dt*Ndt[k])
                eff[eff>graph_limit]=float("nan")  # to remove points of the graph larger than the limit
                lin,=ax.plot(degree,np.log10(eff),linewidth=line_width,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
                ax.scatter(degree,np.log10(eff), linewidth=line_width,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
                ax.plot([],[],label=names_graph[i],color=color[i],linewidth=line_width,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
                count_high_order+=1
                # plt.plot(degree,max_dt,label=methods[i],linewidth=2)
        y_ticks_position=np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],7)
        y_ticks_labels=np.char.add([r"%.1f$\cdot$" % pow(10,math.modf(x)[0]) for x in y_ticks_position],[r"$10^{%.0f}$" % int(math.modf(x)[1]) for x in y_ticks_position])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yticks(y_ticks_position)
        ax.set_yticklabels(y_ticks_labels)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(top=np.minimum(graph_limit,ax.get_ylim()[1]),bottom=np.maximum(0,ax.get_ylim()[0]))
        # plt.legend(fontsize=20)
        plt.ylabel(r'$N^{diss}_{op}$',fontsize=22)
        plt.xlabel('# Stages (MVOs)',fontsize=20)
        plt.subplots_adjust(left=0.24, bottom=0.15, right=0.95, top=0.95)
        if 'dissipation1' in results:
            plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_dx_'+str(dx)+'_f0_'+str(f0)+'_dissipation1_eff.pdf')
        elif 'dissipation2' in results:
            plt.savefig('Dispersion_S_images/'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_dx_'+str(dx)+'_f0_'+str(f0)+'_dissipation2_eff.pdf')
        plt.show()


def graph_methods_dt_max(methods,degree,Ndt,dx=0.005,tol=1e-5,fig_ind='',f0=15,free_surf=1):
    # graphics of the maximum \Delta t allowed by methods and approximation degrees at a given numerical experiment

    # INPUT:
    # methods: the different methods we will consider (vector string)
    # example: selection of the wave equation parameters (string)
    # ord: spatial discretization order ('4','8')
    # equ: equation formulation (scalar, scalar_dx2, elastic)
    # ind_source: indicator of the wave equations' source term treatment ('H_amplified', 'FA_ricker')
    # max_degree: maximum polynomial degree used to graphic (int)
    # dx: spatial discretization step size (float)
    # ind: string used to put in the figure name, for saving
    # Ndt: amount of time-step sizes used to compute the solutions (int)

    # if meth_ind<10:
    #     sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'.npy')
    #     np.save('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy',sol)
    # else:
    #     if method=='FA':
    #         sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_H_amplified_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'.npy')
    #     else:
    #         sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'.npy')
    #     np.save('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy',sol)

    sol_ref=np.load('Dispersion_S/RK_ref_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx/4)+'_f0_'+str(f0)+'.npy')
    a,b,nx,ny,X,Y,param,dt,x0,y0=domain_dispersion(dx/4,c=0.2,test=1)
    sol_ref=sol_ref.reshape((ny,nx),order='F')
    sol_ref=sol_ref[1::4,:-1:4]
    # sol_ref=np.expand_dims(sol_ref.flatten('F'),1)
    sol_ref=sol_ref.flatten('F')
    sol_ref=np.load('Dispersion_S/RK_ref_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'.npy')

    color=np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf'])
    # names_prov=np.array(['FA','HORK','KRY','RK9-7','RK3-2','RK4-4','Leap-frog'])
    names_prov=method_label_graph(methods)
    marker=np.array(['D','o','s','v','P','H','D'])
    line=np.array(['-','--','-.'])
    count_high_order=0
    plt.rcParams["figure.figsize"] = [8,6]
    ax=plt.gca()
    for i in range(len(methods)):
        meth_ind,meth_label=method_label(methods[i])
        if meth_ind<10:
            max_dt=0
            for j in range(len(Ndt)):
                sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[j])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy')
                # error=np.max(np.abs(sol-sol_ref))
                error=np.sqrt(np.sum(pow(sol-sol_ref,2))*dx**dim)
                if error>tol:
                    break
                else:
                    max_dt=2*dt*Ndt[j]
            ax.scatter(meth_ind,max_dt, linewidth=2,marker=marker[i],alpha=0.5,color=color[i],s=60,zorder=5*i)
            ax.scatter(meth_ind,max_dt,marker=marker[i],color=color[i],s=30,zorder=5*i)
            ax.scatter([],[],label=names_prov[i],color=color[i],linewidth=2,marker=marker[i],zorder=5*i)
        else:
            max_dt=np.zeros(len(degree))
            for j in range(len(degree)):
                for k in range(len(Ndt)):
                    if methods[i]=='FA':
                        try:
                            sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_'+ind_source+'_Ndt_'+str(Ndt[k])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy')
                        except:
                            continue
                    else:
                        try:
                            sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_Ndt_'+str(Ndt[k])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_f0_'+str(f0)+'.npy')
                        except:
                            continue
                    # error=np.max(np.abs(sol-sol_ref))
                    error=np.sqrt(np.sum(pow(sol-sol_ref,2))*dx**dim)
                    if error>tol:
                        break
                    else:
                        max_dt[j]=2*dt*Ndt[k]
            lin,=ax.plot(degree,max_dt,linewidth=2,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
            ax.scatter(degree,max_dt, linewidth=2,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
            ax.plot([],[],label=names_prov[i],color=color[i],linewidth=2,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
            count_high_order+=1

    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.legend(fontsize=20)
    plt.ylabel('$\Delta t_{max}$',fontsize=20)
    plt.xlabel('# Stages',fontsize=20)
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.95)
    plt.savefig('Dispersion_S_images/'+example+'_methods_max_dt'+fig_ind+'_equ_scalar_dx2_free_surf_'+str(free_surf)+'_ord_8_dx_'+str(dx)+'.pdf')
    plt.show()