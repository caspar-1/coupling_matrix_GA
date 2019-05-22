import numpy as np
import random
import copy


class GENE_POOL():
    def __init__(self,pool_sz):
        self.pool_sz=pool_sz
        self.pool=[None]*pool_sz
        self.fitest=None

    def initiliase_pool(self,CLS):
        for idx in range(self.pool_sz):
            self.pool[idx]=CLS()

    def breed(self,keep_top_n=5):
        new_population=[None]*self.pool_sz
        for idx in range(self.pool_sz):
            gene_a=random.choice(self.pool)
            gene_b=random.choice(self.pool)
            child=gene_a.breed(gene_b)
            new_population[idx]=child
        fittest_parents = self.pool[0:keep_top_n]
        new_children = new_population[keep_top_n:]
        self.pool = None
        self.pool =  fittest_parents + new_children





    def cull(self,n):
        fitness=[None]*self.pool_sz
        for idx in range(self.pool_sz):
            _gene = self.pool[idx]
            fitness[idx]=_gene.check()

        fit_sorted = np.argsort(fitness)
        print(fit_sorted)
        new_population=[None]*n
        for idx in range(n):
            _i1 = fit_sorted[idx]
            new_population[idx]=self.pool[_i1]
        self.fitest=new_population.copy()
        self.pool=new_population.copy()





class GENE(object):
    sz=0
    mutate_thr=0
    cross_thr=0

    def __init__(self):
        self.gene=((np.random.rand(self.sz)*2)-1)
        self.fitness=None
       

    def breed(self,dna):
        rr=np.random.rand(self.sz)
        r1 = np.where(rr>self.mutate_thr,True,False)
        r2 = np.where(rr<self.cross_thr,True,False)
        child=copy.deepcopy(self)
        debug=["-"]*self.sz
        for _idx in range(self.sz):
            if r1[_idx]:
                rn= (((np.random.rand(1)*2.0)-1.0))
                weight=child.gene[_idx]
                weight_n= weight + (rn/200.0) #random mutate
                child.gene[_idx]=weight_n
                debug[_idx]="M"
            if r2[_idx]:
                #child.gene[_idx]=child.gene[_idx] + (((np.random.rand(1)*2.0)-1.0)) #random mutate
                child.gene[_idx]=dna.gene[_idx] #crossbread
                debug[_idx]="X"

        
        print(''.join(debug))

        return child

    def check(self):
        """get the fitness of a gene, override this method"""
        fitness=0.0
        return fitness



class TEST_GENE(GENE):
    sz=28
    mutate_thr=0.9
    cross_thr=0.1


    def check(self):
        """overide this"""
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
        f=(u+u.T+d)
        np.set_printoptions(precision=3)
        print(f)
        pass




if __name__=="__main__":
    _pool = GENE_POOL(100)
    _pool.initiliase_pool(TEST_GENE)
    _pool.cull(10)
    _pool.breed()

    pass
