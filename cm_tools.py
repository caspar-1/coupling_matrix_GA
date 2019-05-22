import numpy as np
import matplotlib.pyplot as plt

#https://raw.githubusercontent.com/zinka/arraytool/master/arraytool/src/arraytool/filtertool.py



def cutoff(F, dB_limit=-40):
    r"""
    When magnitude of S11 or S21 is 0 in linear scale, their dB value is '-infinity'.
    So, this function will be used to cut-off all the value below some 'dB_limit'.

    :param F:           F is a Numpy array
    :param dB_limit:    cut-off level in dB, default value is -40

    :rtype:             a Numpy array, same size as the input array F
    """
    msk1 = F < dB_limit
    fill = msk1 * dB_limit
    msk2 = F >= dB_limit
    F = F * (msk2) + fill
    return F


def MN_to_Sparam(M, Rs, Rl, w_min=-2, w_max=2, w_num=500, dB=True,
                 dB_limit=-40, plot=True,Fc=None):
    r"""
    Function to plot S parameters from a given (N,N) coupling matrix.

    :param M:      NbyN coupling matrix
    :param Rs:     Source resistance
    :param Rl:     Load resistance
    :param w_min:  lower limit of w (for plotting) 
    :param w_max:  upper limit of w (for plotting) 
    :param w_num:  number of points between 'x_min' and 'x_max' including x_min and x_max
    :param dB:     If true plotting will be done in dB scale
    :param dB_limit:  cut-off level in dB, default value is -40
    :param plot:   If True, plot will be drawn ... but will be showed only if show = True

    :rtype:        [w, S11, S21] ... frequency and the corresponding S11 and S21
                   all in linear scale   
    """
    w = np.linspace(w_min, w_max, w_num)
    R = np.zeros_like(M)
    R[0, 0] = Rs
    R[-1, -1] = Rl
    MR = M - 1j * R
    I = np.eye(M.shape[0], M.shape[1])
    # Calculating S parameters
    S11 = np.zeros((len(w), 1), dtype=complex)
    S21 = np.zeros((len(w), 1), dtype=complex)  # 'dtype' is important
    for i in range(len(w)):
        A = MR + w[i] * I
        A_inv = np.linalg.inv(A)
        S11[i] = 1 + 2j * Rs * A_inv[0, 0]
        S21[i] = -2j * np.sqrt(Rs * Rl) * A_inv[-1, 0]

    if Fc:
        f_min=Fc+((w_min/(2.0*np.pi))*Fc)
        f_max=Fc+((w_max/(2.0*np.pi))*Fc)
        w = np.linspace(f_min, f_max, w_num) 

    if(plot):  # Plotting
        # Converting the S parameters into either linear or dB scale
        if(dB):
            S11_plt = 20 * np.log10(abs(S11))
            S21_plt = 20 * np.log10(abs(S21))
            S11_plt = cutoff(S11_plt, dB_limit)
            S21_plt = cutoff(S21_plt, dB_limit)
            y_labl = r'$\ \mathrm{(dB)}$'
        else:
            S11_plt = abs(S11)
            S21_plt = abs(S21)
            y_labl = r'$\ \mathrm{(linear)}$'
        plt.plot(w, S21_plt, 'b-', label=r"$S_{21}$")
        plt.plot(w, S11_plt, 'r-', label=r"$S_{11}$")
        plt.axis('tight')
        plt.grid(True)
        plt.legend()
        if Fc:
            plt.xlabel(r'$ Hz}$', fontsize=14)
        else:
            plt.xlabel(r'$\Omega\ \mathrm{(rad/s)}$', fontsize=14)
        plt.ylabel(r'$\mathrm{Magnitude}$' + y_labl, fontsize=14)
        plt.show()
    return w, S11, S21


def MN2_to_Sparam(M, Rs=1, Rl=1, w_min=-2, w_max=2, w_num=500, dB=True,
                  dB_limit=-40, plot=True):
    r"""
    Function to plot S parameters from a given (N+2,N+2) coupling matrix.

    :param M:      NbyN coupling matrix
    :param Rs:     Source resistance
    :param Rl:     Load resistance
    :param w_min:  lower limit of w (for plotting) 
    :param w_max:  upper limit of w (for plotting) 
    :param w_num:  number of points between 'x_min' and 'x_max' including x_min and x_max
    :param dB:     If true plotting will be done in dB scale
    :param dB_limit:  cut-off level in dB, default value is -40
    :param plot:   If True, plot will be drawn ... but will be showed only if show = True

    :rtype:    [w, S11, S21] ... frequency and the corresponding S11 and S21
               all in linear scale
    """
    w = np.linspace(w_min, w_max, w_num)
    R = np.zeros_like(M)
    R[0, 0] = Rs
    R[-1, -1] = Rl
    MR = M - 1j * R
    I = np.eye(M.shape[0], M.shape[1])
    I[0, 0] = 0
    I[-1, -1] = 0
    # Calculating S parameters
    S11 = np.zeros((len(w), 1), dtype=complex)
    S21 = np.zeros((len(w), 1), dtype=complex)  # 'dtype' is important
    for i in range(len(w)):
        A = MR + w[i] * I
        A_inv = np.linalg.inv(A)
        S11[i] = 1 + 2j * Rs * A_inv[0, 0]
        S21[i] = -2j * np.sqrt(Rs * Rl) * A_inv[-1, 0]
    if(plot):  # Plotting
        # Converting the S parameters into either linear or dB scale
        if(dB):
            S11_plt = 20 * np.log10(abs(S11))
            S21_plt = 20 * np.log10(abs(S21))
            S11_plt = cutoff(S11_plt, dB_limit)
            S21_plt = cutoff(S21_plt, dB_limit)
            y_labl = r'$\ \mathrm{(dB)}$'
        else:
            S11_plt = abs(S11)
            S21_plt = abs(S21)
            y_labl = r'$\ \mathrm{(linear)}$'
        plt.plot(w, S21_plt, 'b-', label=r"$S_{21}$")
        plt.plot(w, S11_plt, 'r-', label=r"$S_{11}$")
        plt.axis('tight')
        plt.grid(True)
        plt.legend()
        plt.xlabel(r'$\Omega\ \mathrm{(rad/s)}$', fontsize=14)
        plt.ylabel(r'$\mathrm{Magnitude}$' + y_labl, fontsize=14)
        plt.show()
    return w, S11, S21

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))




if __name__ == "__main__":
    cm = np.array([
        [0, 0.0091, 0, 0],
        [0.0091, 0, 0.0070, 0],
        [0, 0.0070, 0, 0.0091],
        [0, 0, 0.0091, 0]])

    cm = np.array([[0.021099,   0.213737,   -0.331426,  0.565696,   -0.273082,  -0.488936,  0.0],
                   [0.213737,   0.914809,   -0.199231,  0.0,        0.0,        -0.199894,  -0.213737],
                   [-0.331426,  -0.199231,  -0.381081,  -0.0,       -0.0,       0.455754,   0.331426],
                   [0.565696,   0.0,        -0.0,       0.362462 ,  0.312493,   0.0,        0.565696],
                   [-0.273082,  0.0,        -0.0,       0.312493,   -0.621957,  0.0,        -0.273082],
                   [-0.488936,  -0.199894,  0.455754,   0.0,        0.0,        -0.712529,  0.488936],
                   [0.0,        -0.213737,  0.331426,   0.565696,   -0.273082,  0.488936,   0.021099]])


    pool = [((np.random.rand(7,7)-1.0)*2.0) for _ in range(100)]


    target_w,target_S11, target_S21 = MN_to_Sparam(cm, Rs=1, Rl=1, w_min=-5, w_max=5, w_num=64, dB=True,dB_limit=-200, plot=False)

    fitness=[0]*100

    for _ in range(200):
        for idx in range(len(pool)):
            w, S11, S21 = MN_to_Sparam(pool[idx], Rs=1, Rl=1, w_min=-5, w_max=5, w_num=64, dB=True,dB_limit=-200, plot=False)
            err_s21 = dist(target_S21,S21)
            #err_s11 = dist(target_S11,S11)
            #fitness[idx]=np.absolute(dist(err_s21,err_s11))
            fitness[idx]=err_s21
        
        fit_sorted = np.argsort(fitness)

        new_pool=[0]*100
        cnt=0
        for idx in fit_sorted[0:10]:
            for idx_i in range(10):
                a = np.ndarray.flatten(pool[idx_i])
                b = np.ndarray.flatten(pool[cnt])
                rr=np.random.rand(49)
                r1 = np.where(rr>0.8,True,False)
                r2 = np.where(rr<0.05,True,False)
                for _i2 in range(len(r1)):
                    if r1[_i2]:
                        a[_i2]=b[_i2] #crossbread
                    if r2[_i2]:
                        a[_i2]=(np.random.rand(1)-1.0)*2.0 #random mutate
               
                new_pool[cnt]= np.reshape(a,[7,7])
                
                cnt+=1

        pool=new_pool
            

    MN_to_Sparam(pool[fit_sorted[0]], Rs=1, Rl=1, w_min=-5, w_max=5, w_num=1024, dB=True,dB_limit=-200, plot=True)


    pass
