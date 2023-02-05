import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import sparse
from scipy.sparse import linalg
plt.rcParams["figure.figsize"] = (12,8)

def parameters(setting):
    if setting == 'tunnel':
        #params= [N_x,dx,N_t,dt,V_0,c,k,frame_skip,interval]
        params = [3000,3e-4,1000,1e-6,6e4,0.0001,0.09,1,1]
        N_x = params[0]
        V_0 = params[4]
        c = params[5]
        k = params[6]
        psi_0 = np.array([1/np.sqrt(2)*np.exp(-c*(i-N_x/3)**2+k*1j*(i-N_x/3)) for i in range(N_x)])
        V = np.zeros((N_x),complex)
        for i in range(N_x):
            if i>0.55*N_x and i<0.59*N_x:
                V[i]=V_0
        params.append(psi_0)
        params.append(V)
        return params

    elif setting == 'double_tunnel':
        #params= [N_x,dx,N_t,dt,V_0,c,k,frame_skip,interval]
        params = [3000,3e-4,1000,1e-6,6e4,0.0001,0.09,1,1]
        N_x = params[0]
        V_0 = params[4]
        c = params[5]
        k = params[6]
        psi_0 = np.array([1/np.sqrt(2)*np.exp(-c*(i-N_x/3)**2+k*1j*(i-N_x/3)) for i in range(N_x)])
        V = np.zeros((N_x),complex)
        for i in range(N_x):
            if i>0.5*N_x:
                if i%100 >=0 and i%100<=40:
                    V[i]=V_0

        params.append(psi_0)
        params.append(V)
        return params

    elif setting == 'step':
        #params= [N_x, dx,  N_t, dt,  V_0,c,     k,   frame_skip,interval]
        params = [3000,3e-4,1000,1e-6,1,0.0001,0.09,1,1]
        N_x = params[0]
        V_0 = params[4]
        c = params[5]
        k = params[6]
        psi_0 = np.zeros((N_x),complex)
        for i in range(N_x):
            if i > 0.45*N_x and i < 0.55*N_x:
                psi_0[i]=0.5+0.5j
        V = np.zeros((N_x),complex)
        
        params.append(psi_0)
        params.append(V)
        return params

    elif setting == 'collision':
        #params= [N_x,dx,N_t,dt,V_0,c,k,frame_skip,interval]
        params = [3000,3e-4,2000,1e-6,6e4,0.0001,0.09,1,1]
        N_x = params[0]
        V_0 = params[4]
        c = params[5]
        k = params[6]
        psi_0 = np.array([1/np.sqrt(2)*np.exp(-c*(i-N_x/4)**2+k*1j*(i-N_x/4)) for i in range(N_x)])
        psi_2 = np.array([1/np.sqrt(2)*np.exp(-c*(i-3*N_x/4)**2-k*1j*(i-3*N_x/4)) for i in range(N_x)])
        psi_0 = psi_0 + psi_2
        V = np.zeros((N_x),complex)
        
        params.append(psi_0)
        params.append(V)
        return params

    if setting == 'circle':
        #params= [N_x,dx,N_t,dt,V_0,c,k,frame_skip,interval]
        params = [3000,3e-4,1000,1e-6,8e4,0.0001,0.09,1,1]
        N_x = params[0]
        V_0 = params[4]
        c = params[5]
        k = params[6]
        psi_0 = np.array([1/np.sqrt(2)*np.exp(-c*(i-N_x/3)**2+k*1j*(i-N_x/3)) for i in range(N_x)])
        V = np.zeros((N_x),complex)

        for i in range(N_x):
            temp = 1-0.05/N_x*(i-0.6*N_x)**2
            if temp>=0:
                V[i]=V_0*np.sqrt(temp)
        
        
        params.append(psi_0)
        params.append(V)
        return params

    if setting == 'delta':
        #params= [N_x, dx,  N_t, dt,  V_0,c,     k,frame_skip,interval]
        params = [3000,3e-4,1000,1e-6,8e4,0.1,0.09,1,10]
        N_x = params[0]
        V_0 = params[4]
        c = params[5]
        k = params[6]
        psi_0 = np.array([3/np.sqrt(2)*np.exp(-c*(i-N_x/3)**2+k*1j*(i-N_x/3)) for i in range(N_x)])
        V = np.zeros((N_x),complex)
        
        params.append(psi_0)
        params.append(V)
        return params

    if setting == 'gauss':
        #params= [N_x, dx,  N_t, dt,  V_0,c,     k,frame_skip,interval]
        params = [3000,3e-4,10000,2e-6,6e4,0.0001,0*0.01,3,1]
        N_x = params[0]
        V_0 = params[4]
        c = params[5]
        k = params[6]
        psi_0 = np.array([1/np.sqrt(2)*np.exp(-c*(i-N_x/2)**2+k*1j*(i-N_x/2)) for i in range(N_x)])
        V = np.zeros((N_x),complex)
        
        params.append(psi_0)
        params.append(V)
        return params
    
    if setting == 'standing_wave':
        #params= [N_x, dx,  N_t, dt,  V_0,c,     k,frame_skip,interval]
        params = [3000,3e-4,10000,2e-6,6e4,0.0001,0.01,3,1]
        N_x = params[0]
        V_0 = params[4]
        c = params[5]
        k = params[6]
        n = 8
        psi_0 = np.array([1/2*np.sin(n*i*np.pi/N_x) for i in range(N_x)])
        V = np.zeros((N_x),complex)
        
        params.append(psi_0)
        params.append(V)
        return params

    else:
        print("Enter valid setting.")
        return 0


def compute_time_evolution(params):
    N_x,dx,N_t,dt,V_0,c,k,frame_skip,interval,psi_0,V = params

    o = np.ones((N_x),complex)
    alpha = (1j)*dt/(2*dx**2)*o
    xi = o+1j*dt/2*(2/(dx**2)*o+V)
    gamma = o-1j*dt/2*(2/(dx**2)*o+V)

    diags = np.array([-1,0,1])
    vecs1 = np.array([-alpha,xi,-alpha])
    vecs2 = np.array([alpha,gamma,alpha])

    U1 = sparse.spdiags(vecs1,diags,N_x,N_x)
    U2 = sparse.spdiags(vecs2,diags,N_x,N_x)
    U1 = U1.tocsc()
    U2 = U2.tocsc()

    psi = np.zeros((N_x,N_t),complex)
    psi[:,0]=psi_0
    LU = linalg.splu(U1)

    for i in range(0,N_t-1):
        b = U2.dot(psi[:,i])
        psi[:,i+1]=LU.solve(b)
        psi[:,i+1][0]=0
        psi[:,i+1][-1]=0

    return psi,V


setting = 'tunnel'
params = parameters(setting)
N_x,dx,N_t,dt,V_0,c,k,frame_skip,interval,psi_0,V = params
psi,V = compute_time_evolution(params)
psi_squared = 2*np.abs(psi)
x = np.array([i*dx for i in range(N_x)])

#Animation
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [],label=r"Re($\psi(x)$)")
ln2, = ax.plot([], [],label=r"Im($\psi(x)$)",color="red")
ln3, = ax.plot([], [],label=r"$psi^2(\psi(x)$)",color="purple")


def init():
    ax.set_xlim(0, N_x*dx)
    ax.set_ylim(-2, 2)
    return ln,ln2,ln3

def update(frame):
    #xdata.append(frame)
    xdata = x
    ydata = np.real(psi[:,frame_skip*frame])
    ln.set_data(xdata, ydata)

    ydata2 = np.imag(psi[:,frame_skip*frame])
    ln2.set_data(xdata, ydata2)

    ydata3 = psi_squared[:,frame_skip*frame]
    ln3.set_data(xdata, ydata3)
    return ln,ln2,ln3

all_frames = np.arange(0,N_t-1)
use_frames = all_frames[::frame_skip]

ani = FuncAnimation(fig, update, frames=use_frames,
                    init_func=init, blit=True, interval = interval)
plt.legend()
plt.plot(x,1/V_0*np.real(V),color="black")
plt.show()