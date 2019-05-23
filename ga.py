import numpy as np
import random
import copy
from abc import ABC, abstractmethod

class GENE_POOL(ABC):
    def __init__(self,**kwargs):
        pool_sz = kwargs.get("pool_sz",100)
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
            child=gene_a.breed(mating_chromzone=gene_b)
            new_population[idx]=child
        fittest_parents = self.pool[0:keep_top_n]

        for x in fittest_parents:
            x.last_mutation = np.zeros(x.sz)
            x.mutation_thr_mod = np.ones(x.sz)

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





class GENE(ABC):
    sz=0
    mutate_thr=0
    cross_thr=0
    scaler=400

    def __init__(self,**kwargs):
        self.gene= ((np.random.rand(self.sz)*0.1)-0.05)
        
        self.last_mutation = np.zeros(self.sz)
        self.mutation_thr_mod = np.ones(self.sz)
        self.fitness=None
       

    def breed(self,**kwargs):
        mating_chromzone=kwargs.get("mating_chromzone")
        rr=np.random.rand(self.sz)
        r1 = np.where(rr>(self.mutate_thr*self.mutation_thr_mod),True,False)
        r2 = np.where(rr<self.cross_thr,True,False)
        child=copy.deepcopy(self)
        debug=["-"]*self.sz
        for _idx in range(self.sz):
            if r1[_idx]:
                debug[_idx]="M"
                child.gene[_idx]=self.mutate(_idx)
            if r2[_idx]:
                debug[_idx]="X"
                child.gene[_idx] = self.crossover(_idx,mating_chromzone)

        
        #print("{0} <{1}>".format(''.join(debug),self.fitness))

        return child

    def check(self):
        """get the fitness of a gene, override this method"""
        fitness=0.0
        return fitness

    @abstractmethod
    def mutate(self,gene_idx):
        """override this"""
        pass
        

    @abstractmethod
    def crossover(self,gene_idx,mating_chromzone,**kwargs):
        """override this"""
        pass


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
    _pool = GENE_POOL(pool_sz=100)
    _pool.initiliase_pool(TEST_GENE)
    _pool.cull(10)
    _pool.breed()

    pass
