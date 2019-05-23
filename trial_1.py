import numpy as np
import cm_tools
import ga
import matplotlib.pyplot as plt


points=64
min_w=-3
max_w=3


class PLOTTER():
    def __init__(self):
        self.fig=None
        self.ax=None
        self.line_s21=None
        self.line_s11=None
        self.line_s21_target=None
        self.line_s11_target=None

    def reset(self):
        self.fig = None
    
    def do_plot(self,w,S11,S21,dB=True) :
        
        def convert(data,lim=-80):
            _plt = 20 * np.log10(abs(data))
            _plt = cm_tools.cutoff(_plt, lim)
            return _plt
        




        S11_plt = convert(S11)
        S21_plt = convert(S21)
        target_S11_plt = convert(target_S11_raw)
        target_S21_plt = convert(target_S21_raw)
        y_labl = r'$\ \mathrm{(dB)}$'
        

        if self.fig is None:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.line_s21, = self.ax.plot(w, S21_plt, 'b-', label=r"$S_{21}$") # Returns a tuple of line objects, thus the comma
            self.line_s11, = self.ax.plot(w, S11_plt, 'r-', label=r"$S_{11}$") # Returns a tuple of line objects, thus the comma
            self.line_s21_target, = self.ax.plot(w, target_S21_plt, 'b--', label=r"$S_{21}$") # Returns a tuple of line objects, thus the comma
            self.line_s11_target, = self.ax.plot(w, target_S11_plt, 'r--', label=r"$S_{11}$") # Returns a tuple of line objects, thus the comma

            plt.axis('tight')
            plt.grid(True)
            plt.legend()
            plt.xlabel(r'$\Omega\ \mathrm{(rad/s)}$', fontsize=14)
            plt.ylabel(r'$\mathrm{Magnitude}$' + y_labl, fontsize=14)
            plt.show()
        else:
            #self.ax.clear()
            self.line_s21.set_ydata(S21_plt)
            self.line_s11.set_ydata(S11_plt)
            self.line_s21_target.set_ydata(target_S21_plt)
            self.line_s11_target.set_ydata(target_S11_plt)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



def dist(x,y):   
    err= np.sqrt(np.sum((x-y)**2))
    #err= np.sum((np.absolute(x)-np.absolute(y))**2)
    return err


class CM_GENE(ga.GENE):
    sz=28
    mutate_thr=0.9
    cross_thr=0.05
    sparse=np.array([
        [1,1,1,1,1,1,0],
        [1,1,1,0,0,1,1],
        [1,1,1,0,0,1,1],
        [1,0,0,1,1,0,1],
        [1,0,0,1,1,0,1],
        [1,1,1,0,0,1,1],
        [0,1,1,1,1,1,1],
    ])

    def __init__(self):
        super(CM_GENE, self).__init__()
        

    def get_cm(self):
        sz=7
        col_st=0
        m=np.zeros((sz,sz))

        gene_idx=0
        for r_idx in range(sz):
            for c_idx in range(col_st,sz):
                m[r_idx][c_idx]=self.gene[gene_idx]
                gene_idx+=1
            col_st+=1

        u=np.triu(m,k=1)
        d=d=np.identity(sz)*m
        cm=(u+u.T+d)
        np.set_printoptions(precision=3)
        cm_sparse = CM_GENE.sparse*cm

        return cm_sparse

    def check(self):
        """overide this"""
        f=self.get_cm()
        w, S11, S21 = cm_tools.MN_to_Sparam(f, Rs=1, Rl=1, w_min=min_w, w_max=max_w, w_num=points, dB=True,dB_limit=-200, plot=False)
        S11=20 * np.log10(abs(S11))
        S21=20 * np.log10(abs(S21))
        err_s21 = dist(target_S21,S21)
        err_s11 = dist(target_S11,S11)
        #self.fitness=np.absolute(err_s21) + np.absolute(err_s11)
        #self.fitness=np.absolute(err_s11)
        self.fitness=np.absolute(err_s21)        
        return self.fitness


    def mutate(self,gene_idx):
        mutaion_scaler = (self.fitness/CM_GENE.scaler)
        rm= (((np.random.rand(1)*2.0)-1.0))* mutaion_scaler
        
        if(self.last_mutation[gene_idx]==0.0):
            gene = self.gene[gene_idx] + rm #random mutate
            self.last_mutation[gene_idx]=rm
            self.mutation_thr_mod[gene_idx]=1.0
        else:
            gene = self.gene[gene_idx]+ self.last_mutation[gene_idx]
            self.mutation_thr_mod[gene_idx]=self.mutation_thr_mod[gene_idx]*0.9
        return gene
       
        

    def crossover(self,gene_idx,mating_chromzone,**kwargs):
        self.last_mutation[gene_idx]=0.0
        new_gene=mating_chromzone.gene[gene_idx] #crossbread
        return new_gene


def generate_target():
    cm = np.array([[0.021099,   0.213737,   -0.331426,  0.565696,   -0.273082,  -0.488936,  0.0],
                   [0.213737,   0.914809,   -0.199231,  0.0,        0.0,        -0.199894,  -0.213737],
                   [-0.331426,  -0.199231,  -0.381081,  -0.0,       -0.0,       0.455754,   0.331426],
                   [0.565696,   0.0,        -0.0,       0.362462 ,  0.312493,   0.0,        0.565696],
                   [-0.273082,  0.0,        -0.0,       0.312493,   -0.621957,  0.0,        -0.273082],
                   [-0.488936,  -0.199894,  0.455754,   0.0,        0.0,        -0.712529,  0.488936],
                   [0.0,        -0.213737,  0.331426,   0.565696,   -0.273082,  0.488936,   0.021099]])

    return  cm_tools.MN_to_Sparam(cm, Rs=1, Rl=1, w_min=min_w, w_max=max_w, w_num=points, dB=True,dB_limit=-200, plot=False)





if __name__=="__main__":
    _plotter=PLOTTER()

    target_w,target_S11_raw, target_S21_raw = generate_target()

    target_S11 = 20 * np.log10(abs(target_S11_raw))
    target_S21 = 20 * np.log10(abs(target_S21_raw))

    _pool = ga.GENE_POOL(pool_sz=50)
    _pool.initiliase_pool(CM_GENE)
    for x in range(3000):
        # if x==50:
        #     points=16
        #     target_w,target_S11_raw, target_S21_raw = generate_target()
        #     target_S11 = 20 * np.log10(abs(target_S11_raw))
        #     target_S21 = 20 * np.log10(abs(target_S21_raw))
        #     _plotter.reset()
        # if x==250:
        #     points=32
        #     target_w,target_S11_raw, target_S21_raw = generate_target()
        #     target_S11 = 20 * np.log10(abs(target_S11_raw))
        #     target_S21 = 20 * np.log10(abs(target_S21_raw))
        #     _plotter.reset()
        # if x==450:
        #     points=64
        #     target_w,target_S11_raw, target_S21_raw = generate_target()
        #     target_S11 = 20 * np.log10(abs(target_S11_raw))
        #     target_S21 = 20 * np.log10(abs(target_S21_raw))
        #     _plotter.reset()
        
        _pool.cull(10)
        _pool.breed(keep_top_n=3)
        if x%5==0:
            best=_pool.fitest[0]
            top_cm=best.get_cm()
            target_w,meas_S11, meas_S21=cm_tools.MN_to_Sparam(top_cm, Rs=1, Rl=1, w_min=min_w, w_max=max_w, w_num=points, dB=True,dB_limit=-200, plot=False)
            _plotter.do_plot(target_w,meas_S11, meas_S21)
            print(top_cm)


    input("press key")