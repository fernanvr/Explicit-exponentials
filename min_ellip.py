import numpy as np
from graph_q_ellip import *


def g_1_sign(a12,a22,b1,b2,c):

    return np.sign(a22*pow(b1,2)-2*b1*b2*a12+pow(b2,2)-4*c*(a22-pow(a12,2)))


def g(a12,a22,b1,c,Q):

    return ((1+a22)*pow(a22-pow(a12,2),-2)+2*pow(a22-pow(a12,2),-3/2))*(a22*pow(b1,2)+2*b1*a12*(Q[2,1]*a22+c/Q[2,1])+pow(a22*Q[2,1]+c/Q[2,1],2)-4*c*(a22-pow(a12,2)))/4


def g_4(a12,b1,c,Q):

    r0=-(pow(Q[3,0],2)+b1*Q[3,0]+c*(1-Q[3,1]/Q[2,1]))/(pow(Q[3,1],2)-Q[3,1]*Q[2,1])
    r1=-2*Q[3,0]*Q[3,1]/(pow(Q[3,1],2)-Q[3,1]*Q[2,1])
    a22=r1*a12+r0

    return ((1+a22)*pow(a22-pow(a12,2),-2)+2*pow(a22-pow(a12,2),-3/2))*(a22*pow(b1,2)+2*b1*a12*(Q[2,1]*a22+c/Q[2,1])+pow(a22*Q[2,1]+c/Q[2,1],2)-4*c*(a22-pow(a12,2)))/4


def diff_g_3(a12,a22,b1,c,Q):

    Dg=np.zeros((2,1))

    aux1=a22*pow(b1,2)+2*b1*a12*(Q[2,1]*a22+c/Q[2,1])+pow(a22*Q[2,1]+c/Q[2,1],2)-4*c*(a22-pow(a12,2))
    aux2=(1+a22)+2*pow(a22-pow(a12,2),1/2)

    Dg[0,0]=(4*a12*(1+a22)*pow(a22-pow(a12,2),-1)+6*a12*pow(a22-pow(a12,2),-1/2))*aux1+aux2*((a22*Q[2,1]+c/Q[2,1])*2*b1+8*c*a12)

    Dg[1,0]=-((a22+2+pow(a12,2))*pow(a22-pow(a12,2),-1)+3*pow(a22-pow(a12,2),-1/2))*aux1+aux2*(2*b1*Q[2,1]*a12+2*a22*pow(Q[2,1],2)+pow(b1,2)-2*c)

    return Dg*pow(a22-pow(a12,2),-2)/4


def diff_g_4(a12,b1,c,Q):

    r0=-(pow(Q[3,0],2)+b1*Q[3,0]+c*(1-Q[3,1]/Q[2,1]))/(pow(Q[3,1],2)-Q[3,1]*Q[2,1])
    r1=-2*Q[3,0]*Q[3,1]/(pow(Q[3,1],2)-Q[3,1]*Q[2,1])
    a22=r1*a12+r0

    aux1=a22*pow(b1,2)+2*b1*a12*(Q[2,1]*a22+c/Q[2,1])+pow(a22*Q[2,1]+c/Q[2,1],2)-4*c*(a22-pow(a12,2))
    aux2=(1+a22)+2*pow(a22-pow(a12,2),1/2)

    # print('condition: ',a22-pow(a12,2))

    Dg=((r1*(a22-pow(a12,2))-2*(1+a22)*(r1-2*a12))*pow(a22-pow(a12,2),-1)-3*(r1-2*a12)*pow(a22-pow(a12,2),-1/2))*aux1\
            +aux2*(r1*pow(b1,2)+4*b1*a12*r1*Q[2,1]+2*b1*(Q[2,1]*r0+c/Q[2,1])+2*Q[2,1]*r1*(Q[2,1]*r1*a12+Q[2,1]*r0+c/Q[2,1])-4*c*(r1-2*a12))

    return Dg*pow(a22-pow(a12,2),-2)/4


def hess_g_3(a12,a22,b1,c,Q):

    Hg=np.zeros((2,2))

    aux1=a22*pow(b1,2)+2*b1*a12*(Q[2,1]*a22+c/Q[2,1])+pow(a22*Q[2,1]+c/Q[2,1],2)-4*c*(a22-pow(a12,2))
    aux2=(1+a22)+2*pow(a22-pow(a12,2),1/2)

    Hg[0,0]=(4*(1+a22)*(a22+5*pow(a12,2))*pow(a22-pow(a12,2),-2)+6*(a22+4*pow(a12,2))*pow(a22-pow(a12,2),-3/2))*aux1
    Hg[0,0]=Hg[0,0]+2*(4*a12*(1+a22)*pow(a22-pow(a12,2),-1)+6*a12*pow(a22-pow(a12,2),-1/2))*((a22*Q[2,1]+c/Q[2,1])*2*b1+8*c*a12)
    Hg[0,0]=Hg[0,0]+aux2*8*c

    Hg[0,1]=-(4*a12*(2*a22+pow(a12,2)+3)*pow(a22-pow(a12,2),-2)+15*a12*pow(a22-pow(a12,2),-3/2))*aux1
    Hg[0,1]=Hg[0,1]+(4*a12*(1+a22)*pow(a22-pow(a12,2),-1)+6*a12*pow(a22-pow(a12,2),-1/2))*(2*b1*Q[2,1]*a12+2*a22*pow(Q[2,1],2)+pow(b1,2)-2*c)
    Hg[0,1]=Hg[0,1]-((a22+2+pow(a12,2))*pow(a22-pow(a12,2),-1)+3*pow(a22-pow(a12,2),-1/2))*((a22*Q[2,1]+c/Q[2,1])*2*b1+8*c*a12)
    Hg[0,1]=Hg[0,1]+aux2*2*b1*Q[2,1]

    Hg[1,0]=-Hg[0,1]

    Hg[1,1]=((2*a22+6+4*pow(a12,2))*pow(a22-pow(a12,2),-2)+15/2*pow(a22-pow(a12,2),-3/2))*aux1
    Hg[1,1]=Hg[1,1]-2*((a22+2+pow(a12,2))*pow(a22-pow(a12,2),-1)+3*pow(a22-pow(a12,2),-1/2))*(2*b1*Q[2,1]*a12+2*a22*pow(Q[2,1],2)+pow(b1,2)-2*c)
    Hg[1,1]=Hg[1,1]+aux2*2*pow(Q[2,1],2)

    aux=Hg[0,0]
    Hg[0,0]=Hg[1,1]
    Hg[1,1]=aux
    Hg[0,1]=-Hg[0,1]

    return Hg/(Hg[0,0]*Hg[1,1]-pow(Hg[0,1],2))*pow(a22-pow(a12,2),2)*4


def hess_g_4(a12,b1,c,Q):

    r0=-(pow(Q[3,0],2)+b1*Q[3,0]+c*(1-Q[3,1]/Q[2,1]))/(pow(Q[3,1],2)-Q[3,1]*Q[2,1])
    r1=-2*Q[3,0]*Q[3,1]/(pow(Q[3,1],2)-Q[3,1]*Q[2,1])
    a22=r1*a12+r0

    aux1=a22*pow(b1,2)+2*b1*a12*(Q[2,1]*a22+c/Q[2,1])+pow(a22*Q[2,1]+c/Q[2,1],2)-4*c*(a22-pow(a12,2))
    aux2=((1+a22)+2*pow(a22-pow(a12,2),1/2))

    aux11=(r1*(a22-pow(a12,2))-2*(1+a22)*(r1-2*a12))*pow(a22-pow(a12,2),-1)-3*(r1-2*a12)*pow(a22-pow(a12,2),-1/2)
    aux21=(r1*pow(b1,2)+4*b1*a12*r1*Q[2,1]+2*b1*(Q[2,1]*r0+c/Q[2,1])+2*Q[2,1]*r1*(Q[2,1]*r1*a12+Q[2,1]*r0+c/Q[2,1])-4*c*(r1-2*a12))

    Hg=((pow(r1,2)-2*a12*r1-2*(-4*r1*a12-2*(1+r0)+pow(r1,2))-3*(r1*(a22-pow(a12,2))-2*(1+a22)*(r1-2*a12))*(r1-2*a12)*pow(a22-pow(a12,2),-1))*pow(a22-pow(a12,2),-1)\
        +(6+15/2*pow(r1-2*a12,2)*pow(a22-pow(a12,2),-1))*pow(a22-pow(a12,2),-1/2))*aux1\
       +2*aux11*aux21+aux2*(4*b1*r1*Q[2,1]+2*pow(Q[2,1]*r1,2)+8*c)

    return pow(a22-pow(a12,2),2)*4/Hg


def aff_trans_3(R):
    v1=R[1,:]-R[0,:]
    v2=R[2,:]-R[1,:]
    v3=R[0,:]-R[2,:]

    index=1
    T=np.array([[v3[0],v3[1]],[-v3[1],v3[0]]])/np.sqrt(pow(v3[0],2)+pow(v3[1],2))
    a=R[0,1]-R[2,1]
    b=R[2,0]-R[0,0]
    c=R[0,0]*R[2,1]-R[0,1]*R[2,0]
    x0=(b*(b*R[1,0]-a*R[1,1])-a*c)/(pow(a,2)+pow(b,2))
    y0=(a*(-b*R[1,0]+a*R[1,1])-b*c)/(pow(a,2)+pow(b,2))
    if (x0-R[2,0])*(x0-R[0,0])>pow(10,-16) or (y0-R[2,1])*(y0-R[0,1])>pow(10,-16) or (np.abs(x0-R[0,0])+np.abs(y0-R[0,1]))<pow(10,-10) or (np.abs(x0-R[2,0])+np.abs(y0-R[2,1]))<pow(10,-10):
        index=2
        T=np.array([[v1[0],v1[1]],[-v1[1],v1[0]]])/np.sqrt(pow(v1[0],2)+pow(v1[1],2))
        a=R[0,1]-R[1,1]
        b=R[1,0]-R[0,0]
        c=R[0,0]*R[1,1]-R[0,1]*R[1,0]
        x0=(b*(b*R[2,0]-a*R[2,1])-a*c)/(pow(a,2)+pow(b,2))
        y0=(a*(-b*R[2,0]+a*R[2,1])-b*c)/(pow(a,2)+pow(b,2))
        if (x0-R[1,0])*(x0-R[0,0])>pow(10,-16) or (y0-R[1,1])*(y0-R[0,1])>pow(10,-16) or (np.abs(x0-R[0,0])+np.abs(y0-R[0,1]))<pow(10,-10) or (np.abs(x0-R[1,0])+np.abs(y0-R[1,1]))<pow(10,-10):
            index=0
            T=np.array([[v2[0],v2[1]],[-v2[1],v2[0]]])/np.sqrt(pow(v2[0],2)+pow(v2[1],2))

    aux=T.dot(R[index,:].T)
    u=np.zeros((1,2))
    u[0,0]=-aux[0]
    if index==0:
        aux=T.dot(R[1,:].T)
    else:
        aux=T.dot(R[0,:].T)
    u[0,1]=-aux[1]

    Q=R*0
    if index==0:
        Q[2,:]=(T.dot(R[0,:].T)+u)
        Q[1,:]=(T.dot(R[1,:].T)+u)
        Q[0,:]=(T.dot(R[2,:].T)+u)
    elif index==1:
        Q[0,:]=(T.dot(R[0,:].T)+u)
        Q[2,:]=(T.dot(R[1,:].T)+u)
        Q[1,:]=(T.dot(R[2,:].T)+u)
    else:
        Q[0,:]=(T.dot(R[0,:].T)+u)
        Q[1,:]=(T.dot(R[1,:].T)+u)
        Q[2,:]=(T.dot(R[2,:].T)+u)

    return Q,T,u


def aff_trans_4(R):

    diag1=np.array([0,1])
    diag2=np.array([2,3])
    det1=np.linalg.det(np.array([[R[0,0]-R[2,0],R[1,0]-R[2,0]],[R[0,1]-R[2,1],R[1,1]-R[2,1]]]))
    det2=np.linalg.det(np.array([[R[0,0]-R[3,0],R[1,0]-R[3,0]],[R[0,1]-R[3,1],R[1,1]-R[3,1]]]))
    if np.sign(det1*det2)>0:
        diag1=np.array([0,2])
        diag2=np.array([1,3])
        det1=np.linalg.det(np.array([[R[0,0]-R[1,0],R[2,0]-R[1,0]],[R[0,1]-R[1,1],R[2,1]-R[1,1]]]))
        det2=np.linalg.det(np.array([[R[0,0]-R[3,0],R[2,0]-R[3,0]],[R[0,1]-R[3,1],R[2,1]-R[3,1]]]))
        if np.sign(det1*det2)>0:
            diag1=np.array([0,3])
            diag2=np.array([1,2])

    index=diag2[0]  # if 2: the transformation leaves diag1 to x axes, if 1, leaves diag2 to x axis
    v=R[diag1[1],:]-R[diag1[0],:]
    T=np.array([[v[0],v[1]],[-v[1],v[0]]])/np.sqrt(pow(v[0],2)+pow(v[1],2))
    a=R[diag1[0],1]-R[diag1[1],1]
    b=R[diag1[1],0]-R[diag1[0],0]
    c=R[diag1[0],0]*R[diag1[1],1]-R[diag1[0],1]*R[diag1[1],0]
    x01=(b*(b*R[diag2[0],0]-a*R[diag2[0],1])-a*c)/(pow(a,2)+pow(b,2))
    y01=(a*(-b*R[diag2[0],0]+a*R[diag2[0],1])-b*c)/(pow(a,2)+pow(b,2))
    if (x01-R[diag1[1],0])*(x01-R[diag1[0],0])>pow(10,-16) or (y01-R[diag1[1],1])*(y01-R[diag1[0],1])>pow(10,-16) or (np.abs(x01-R[diag1[0],0])+np.abs(y01-R[diag1[0],1]))<pow(10,-10) or (np.abs(x01-R[diag1[1],0])+np.abs(y01-R[diag1[1],1]))<pow(10,-10):
        index=diag2[1]
        x02=(b*(b*R[diag2[1],0]-a*R[diag2[1],1])-a*c)/(pow(a,2)+pow(b,2))
        y02=(a*(-b*R[diag2[1],0]+a*R[diag2[1],1])-b*c)/(pow(a,2)+pow(b,2))
        if (x02-R[diag1[1],0])*(x02-R[diag1[0],0])>pow(10,-16) or (y02-R[diag1[1],1])*(y02-R[diag1[0],1])>pow(10,-16) or (np.abs(x02-R[diag1[0],0])+np.abs(y02-R[diag1[0],1]))<pow(10,-10) or (np.abs(x02-R[diag1[1],0])+np.abs(y02-R[diag1[1],1]))<pow(10,-10):
            index=diag1[0]
            v=R[diag2[1],:]-R[diag2[0],:]
            T=np.array([[v[0],v[1]],[-v[1],v[0]]])/np.sqrt(pow(v[0],2)+pow(v[1],2))
            a=R[diag2[0],1]-R[diag2[1],1]
            b=R[diag2[1],0]-R[diag2[0],0]
            c=R[diag2[0],0]*R[diag2[1],1]-R[diag1[0],1]*R[diag1[1],0]
            x01=(b*(b*R[diag1[0],0]-a*R[diag1[0],1])-a*c)/(pow(a,2)+pow(b,2))
            y01=(a*(-b*R[diag1[0],0]+a*R[diag1[0],1])-b*c)/(pow(a,2)+pow(b,2))
            if (x01-R[diag2[1],0])*(x01-R[diag2[0],0])>pow(10,-16) or (y01-R[diag2[1],1])*(y01-R[diag2[0],1])>pow(10,-16) or (np.abs(x01-R[diag2[0],0])+np.abs(y01-R[diag2[0],1]))<pow(10,-10) or (np.abs(x01-R[diag2[1],0])+np.abs(y01-R[diag2[1],1]))<pow(10,-10):
                index=diag1[1]

    u=np.zeros((1,2))
    u[0,0]=-T.dot(R[index,:].T)[0]
    Q=R*0
    if index in diag1:
        u[0,1]=-T.dot(R[diag2[0],:].T)[1]
        Q[0,:]=T.dot(R[diag2[0],:].T)+u
        Q[1,:]=T.dot(R[diag2[1],:].T)+u
        Q[2,:]=T.dot(R[index,:].T)+u
        Q[3,:]=T.dot(R[np.squeeze(diag1[diag1!=index]),:].T)+u
    else:
        u[0,1]=-T.dot(R[diag1[0],:].T)[1]
        Q[0,:]=T.dot(R[diag1[0],:].T)+u
        Q[1,:]=T.dot(R[diag1[1],:].T)+u
        Q[2,:]=T.dot(R[index,:].T)+u
        Q[3,:]=T.dot(R[np.squeeze(diag2[diag2!=index]),:].T)+u

    return Q,T,u


def aff_trans_inv(a12,a22,b1,b2,c,T,u):

    A=np.array([[1,a12/2],[a12/2,a22]])
    b=np.array([[b1,b2]])

    A1=(T.T).dot(A.dot(T))
    b1=b.dot(T)+2*u.dot(A.dot(T))
    c1=u.dot(A.dot(u.T))+b.dot(u.T)+c

    return ((A1[0,1]+A1[1,0]),A1[1,1],b1[0,0],b1[0,1],c1[0,0])/A1[0,0]


def newton_3(a12,a22,b1,c,Q,e,max_it):

    for i in range(max_it):
        Dg=diff_g_3(a12,a22,b1,c,Q)
        # if i==0:
            # print('Dg0: ', np.max(np.abs(Dg)))
        if np.max(np.abs(Dg))<e:
            break
        Hg=hess_g_3(a12,a22,b1,c,Q)

        aux=-Hg.dot(Dg)
        a12=a12+aux[0]
        a22=np.max(np.array([a22+aux[1],e+pow(a12,2)]))

    # print('Dg: ',np.max(np.abs(Dg)))

    return a12,a22


def newton_4(a12,b1,c,Q,e,max_it,lim_inf,lim_sup):

    for i in range(max_it):
        Dg=diff_g_4(a12,b1,c,Q)
        # if i==0:
        #     print('Dg0: ', np.max(np.abs(Dg)))
        if np.max(np.abs(Dg))<e:
            break
        Hg=hess_g_4(a12,b1,c,Q)

        a12=a12-Hg*Dg
        if a12<lim_inf:
            a12=lim_inf+np.minimum((lim_sup-lim_inf)/100,0.1)
        elif a12>lim_sup:
            a12=lim_sup-np.minimum((lim_sup-lim_inf)/100,0.1)
        # print('i,Dg,a12: ',i,Dg,a12)

    # print('Dg: ',np.max(np.abs(Dg)))

    return a12


def min_el(R,e,max_it):

    a12_0=0
    a22_0=0
    b1_0=0
    b2_0=0
    c_0=0
    if len(R)==3:
        # print('R min_ell',R)
        Q,T,u=aff_trans_3(R)
        c=(Q[0,0]*pow(Q[1,0],2)-Q[1,0]*pow(Q[0,0],2))/(Q[1,0]-Q[0,0])
        b1=-(Q[1,0]+Q[0,0])
        a22=-Q[0,0]*Q[1,0]/pow(Q[2,1],2)
        # b2=0
        a12=0

        # print('Q,T,u', Q,T,u)

        # print('a12,a22,b1,c',a12,a22,b1,c)

        # print('initial condition: ',a12,a22,b1,b2,c)
        # a12_0,a22_0,b1_0,b2_0,c_0=aff_trans_inv(a12,a22,b1,b2,c,T,u)
        # graph_q_ellip(R,a12_0,a22_0,b1_0,b2_0,c_0,100,'g',1)
        # print('square of semi-axes sum: ',g(a12,a22,b1,c,Q))
        #
        a12,a22=newton_3(a12/2,a22,b1,c,Q,e,max_it)
        b2=-a22*Q[2,1]-c/Q[2,1]

        # print('a12,a22,b1,c later',a12,a22,b1,c)

        # R min_ell[[-1.49400600e+01 - 4.51847341e+02]
        # [-2.19735341e-12 4.51847341e+02]
        # [-2.19735341e-12 - 4.51847341e+02]]
        # Q, T, u[[14.94006       0.]
        # [0.  0.]
        # [0. - 903.69468246]] [[-1.  0.]
        #                       [-0. - 1.]][[-2.19735341e-12 - 4.51847341e+02]]
        # a12, a22, b1, c
        # 0 - 0.0 - 14.940059999997803 - 0.0
        # a12, a22, b1, c
        # later[nan]
        # nan - 14.940059999997803 - 0.0

        # print('optimum parameters: ',2*a12,a22,b1,b2,c)
        a12_0,a22_0,b1_0,b2_0,c_0=aff_trans_inv(a12,a22,b1,b2,c,T,u)
        # graph_q_ellip(R,a12_0,a22_0,b1_0,b2_0,c_0,100,'r',1)
        # print('square of semi-axes sum',g(a12,a22,b1,c,Q))
        #
        # q=np.array([1,0])
        # print('condition: ',(pow(q[0],2)+a12_0*q[0]*q[1]+a22_0*pow(q[1],2)+b1_0*q[0]+b2_0*q[1]+c_0)*g_1_sign(a12_0,a22_0,b1_0,b2_0,c_0))
        # plt.show()
    elif len(R)==4:
        # plt.show()
        Q,T,u=aff_trans_4(R)

        c=(Q[0,0]*pow(Q[1,0],2)-Q[1,0]*pow(Q[0,0],2))/(Q[1,0]-Q[0,0])
        b1=-(Q[1,0]+Q[0,0])
        r0=-(pow(Q[3,0],2)+b1*Q[3,0]+c*(1-Q[3,1]/Q[2,1]))/(pow(Q[3,1],2)-Q[3,1]*Q[2,1])
        r1=-Q[3,0]*Q[3,1]/(pow(Q[3,1],2)-Q[3,1]*Q[2,1])
        if np.abs(r1)<pow(10,-14):
            a12=0
            lim_inf=-np.sqrt(r0)
            lim_sup=np.sqrt(r0)
        else:
            lim_inf=2*(r1-np.sqrt(pow(r1,2)+r0))
            lim_sup=2*(r1+np.sqrt(pow(r1,2)+r0))
            if r1>0:
                lim_inf=np.maximum(lim_inf,-r0/r1)
            else:
                lim_sup=np.minimum(lim_sup,-r0/r1)
            a12=(lim_inf+lim_sup)/2

        # a22=r1*a12+r0
        # b2=-(Q[2,1]*a22+c/Q[2,1])

        # print('Q: ',Q)
        #
        # print('r1: ',r1)
        # print('r0: ',r0)
        # print('a22: ',a22)
        # print('a22-a12^2',a22-pow(a12,2)/4)
        #
        # a12_0,a22_0,b1_0,b2_0,c_0=aff_trans_inv(a12,a22,b1,b2,c,T,u)
        # print('initial condition: ',a12_0,a22_0,b1_0,b2_0,c_0)
        # graph_q_ellip(R,a12_0,a22_0,b1_0,b2_0,c_0,10000,'g')

        a12=2*newton_4(a12/2,b1,c,Q,e,max_it,lim_inf/2,lim_sup/2)
        a22=r1*a12+r0
        b2=-(Q[2,1]*a22+c/Q[2,1])
        a12_0,a22_0,b1_0,b2_0,c_0=aff_trans_inv(a12,a22,b1,b2,c,T,u)

        # graph_q_ellip(R,a12_0,a22_0,b1_0,b2_0,c_0,10000,'r')
        # plt.show()
    elif len(R)>=5:
        det0=np.linalg.det(np.array([[R[0,0]*R[0,1],pow(R[0,1],2),R[0,0],R[0,1],1],
                       [R[1,0]*R[1,1],pow(R[1,1],2),R[1,0],R[1,1],1],
                       [R[2,0]*R[2,1],pow(R[2,1],2),R[2,0],R[2,1],1],
                       [R[3,0]*R[3,1],pow(R[3,1],2),R[3,0],R[3,1],1],
                       [R[4,0]*R[4,1],pow(R[4,1],2),R[4,0],R[4,1],1]]))
        a12_0=np.linalg.det(np.array([[-pow(R[0,0],2),pow(R[0,1],2),R[0,0],R[0,1],1],
                       [-pow(R[1,0],2),pow(R[1,1],2),R[1,0],R[1,1],1],
                       [-pow(R[2,0],2),pow(R[2,1],2),R[2,0],R[2,1],1],
                       [-pow(R[3,0],2),pow(R[3,1],2),R[3,0],R[3,1],1],
                       [-pow(R[4,0],2),pow(R[4,1],2),R[4,0],R[4,1],1]]))/det0
        a22_0=np.linalg.det(np.array([[R[0,0]*R[0,1],-pow(R[0,0],2),R[0,0],R[0,1],1],
                       [R[1,0]*R[1,1],-pow(R[1,0],2),R[1,0],R[1,1],1],
                       [R[2,0]*R[2,1],-pow(R[2,0],2),R[2,0],R[2,1],1],
                       [R[3,0]*R[3,1],-pow(R[3,0],2),R[3,0],R[3,1],1],
                       [R[4,0]*R[4,1],-pow(R[4,0],2),R[4,0],R[4,1],1]]))/det0
        b1_0=np.linalg.det(np.array([[R[0,0]*R[0,1],pow(R[0,1],2),-pow(R[0,0],2),R[0,1],1],
                       [R[1,0]*R[1,1],pow(R[1,1],2),-pow(R[1,0],2),R[1,1],1],
                       [R[2,0]*R[2,1],pow(R[2,1],2),-pow(R[2,0],2),R[2,1],1],
                       [R[3,0]*R[3,1],pow(R[3,1],2),-pow(R[3,0],2),R[3,1],1],
                       [R[4,0]*R[4,1],pow(R[4,1],2),-pow(R[4,0],2),R[4,1],1]]))/det0
        b2_0=np.linalg.det(np.array([[R[0,0]*R[0,1],pow(R[0,1],2),R[0,0],-pow(R[0,0],2),1],
                       [R[1,0]*R[1,1],pow(R[1,1],2),R[1,0],-pow(R[1,0],2),1],
                       [R[2,0]*R[2,1],pow(R[2,1],2),R[2,0],-pow(R[2,0],2),1],
                       [R[3,0]*R[3,1],pow(R[3,1],2),R[3,0],-pow(R[3,0],2),1],
                       [R[4,0]*R[4,1],pow(R[4,1],2),R[4,0],-pow(R[4,0],2),1]]))/det0
        c_0=np.linalg.det(np.array([[R[0,0]*R[0,1],pow(R[0,1],2),R[0,0],R[0,1],-pow(R[0,0],2)],
                       [R[1,0]*R[1,1],pow(R[1,1],2),R[1,0],R[1,1],-pow(R[1,0],2)],
                       [R[2,0]*R[2,1],pow(R[2,1],2),R[2,0],R[2,1],-pow(R[2,0],2)],
                       [R[3,0]*R[3,1],pow(R[3,1],2),R[3,0],R[3,1],-pow(R[3,0],2)],
                       [R[4,0]*R[4,1],pow(R[4,1],2),R[4,0],R[4,1],-pow(R[4,0],2)]]))/det0
        # graph_q_ellip(R,a12_0,a22_0,b1_0,b2_0,c_0,100,'g')
    # plt.show()
    return R,a12_0,a22_0,b1_0,b2_0,c_0


def in_el(q,R,a12,a22,b1,b2,c):

    if len(R)==0:
        return 0
    else:
        aux=R-q
        aux=np.sqrt(pow(aux[:,0],2)+pow(aux[:,1],2))
        if np.min(aux)<pow(10,-8):  # condition to avoid consider ellipse with points too close from each other
            return 1
        elif len(R)==1:
            if np.abs(q[0]-R[0,0])>pow(10,-14) or np.abs(q[1]-R[0,1])>pow(10,-14):
                return 0
            else:
                return 1
        elif len(R)==2:
            if np.abs((R[0,1]-R[1,1])*q[0]+(R[1,0]-R[0,0])*q[1]+R[0,0]*R[1,1]-R[1,0]*R[0,1])>pow(10,-14):
                return 0
            elif (q[0]-R[0,0])*(q[0]-R[1,0])>0 or (q[1]-R[0,1])*(q[1]-R[1,1])>0:
                return 0
            else:
                return 1
        else:
            # print('signo: ',(pow(q[0],2)+a12*q[0]*q[1]+a22*pow(q[1],2)+b1*q[0]+b2*q[1]+c)*g_1_sign(a12,a22,b1,b2,c))
            # print('q:', q)
            # print('coefficients: ',a12,a22,b1,b2,c)
            if (pow(q[0],2)+a12*q[0]*q[1]+a22*pow(q[1],2)+b1*q[0]+b2*q[1]+c)*g_1_sign(a12,a22,b1,b2,c)*g_1_sign(a12,a22,b1,b2,c)>0:
                return 0
            else:
                return 1


