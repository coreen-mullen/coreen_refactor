import sys
from GRSlib.Ver0_Files.opt_tools import *

#-----------------------------------------------------------------------

# GSQS local optimization
#-----------------------------------------------------------------------
import jax.numpy as np
import numpy as vnp
from jax import grad, jit, vmap
from jax import random
from functools import partial
import scipy as sp
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import lammps
import lammps.mliap
from lammps.mliap.loader import *
import pandas as pd
import copy
import pickle
import os
import shutil
import random

n_totconfig = 10
data_path = 'bcc.data'
cross_weight =1.000000
self_weight = 1.000000
randomize_comps= False # # flag to use randomized compositions for elements in the dictionary: target_comps = {'Cr':1.0 }
mincellsize = 54
maxcellsize=55
target_comps = {'W:1.0'}
min_typ_global='box' #box or min
soft_strength=0.0
elems=get_desc_count('coupling_coefficients.yace',return_elems=True)
nelements= len(elems)
n_descs= get_desc_count('coupling_coefficients.yace')
rand_comp =1
target_comps = {'W':1.0}
# extra objective function variables to target exact matches of descriptors
q1_wt=5.0
q1_cross_wt=5.0
q2_wt=0.0
q2_cross_wt=0.0
# flag to perform minimization with variable cell lengths
box_rlx = False
# flag to run at temperature first
run_temp = False
# flag to use fire style minimization (good for phase transitions & large minimizations)
fire_min = False
# flag to use linestyle quadratic minimization
line_min = True
# flag to use randomized compositions for elements in the dictionary: target_comps = {'Cr':1.0 }
randomize_comps=True
class GRSModel:
    def __init__(self, n_elements, n_descriptors_tot, mask):
        self.mask=mask
        self.n_descriptors=n_descriptors_tot
        self.n_descriptors_keep=len(mask)*n_elements
        self.n_elements=n_elements
        self.n_params=1
        self.sum_of_products=vnp.zeros((self.n_descriptors_keep,self.n_descriptors_keep))
        self.sum=vnp.zeros((self.n_descriptors_keep,))
        self.sumsq=vnp.zeros((self.n_descriptors_keep,))
        self.q_count=0
        self.first_moment_grad=grad(self.first_moment)
        self.V_grad=grad(self.V)
        self.K_self=self_weight
        self.K_cross=cross_weight
        self.first_mom_weight_cross = q1_cross_wt
        self.first_mom_weight = q1_wt
        self.second_mom_weight_cross = q2_cross_wt
        self.second_mom_weight = q2_wt
        self.whitening=np.identity(self.n_descriptors_keep)
        self.mode=  "update"
        self.data=[]

    def set_mode_update(self):
        self.mode="update"

    def set_mode_run(self):
        self.mode="run"

    def update(self,d):
        if self.n_elements > 1:
            dft=d.flatten()
        else:
            dft = d
        self.q_count += dft.shape[0]
        self.sum+=vnp.sum(dft,axis=0)
        self.sumsq+=vnp.sum(dft*dft,axis=0)
        self.set_mode_run()

    #match for mean descriptor value for the set of structures
    @partial(jit, static_argnums=(0))
    def first_moment_cross(self, descriptors):
        #n_atoms=descriptors.shape[0]
        #n_descriptors=descriptors.shape[1]
        avgs = self.sum/self.q_count
        abs_diffs_1 = np.abs(avgs - descriptors_target_1)
        abs_diffs_1 = np.array(abs_diffs_1)
        abs_diffs_1 = np.nan_to_num(abs_diffs_1)
        is_zero = np.isclose(abs_diffs_1,np.zeros(abs_diffs_1.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.first_mom_weight_cross)
        tst_residual_1 = np.sum(abs_diffs_1) +bonus
        return tst_residual_1

    #match for variance of descriptor value for the set of structures
    @partial(jit, static_argnums=(0))
    def second_moment_cross(self, descriptors):
        #n_atoms=descriptors.shape[0]
        #n_descriptors=descriptors.shape[1]
        vrs = self.sumsq/self.q_count 
        abs_diffs_2 = np.abs(vrs - descriptors_target_2)
        abs_diffs_2 = np.array(abs_diffs_2)
        abs_diffs_2 = np.nan_to_num(abs_diffs_2)
        is_zero = np.isclose(abs_diffs_2,np.zeros(abs_diffs_2.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.second_mom_weight_cross)
        tst_residual_2 = np.sum(abs_diffs_2) +bonus
        #print (tst_residual_2)
        return tst_residual_2

    # match of mean descriptor values within the current structure only
    @partial(jit, static_argnums=(0))
    def first_moment(self, descriptors):
        #n_atoms=descriptors.shape[0]
        #n_descriptors=descriptors.shape[1]
        avgs = np.average(descriptors,axis=0)
        abs_diffs_1 = np.abs(avgs - descriptors_target_1)
        abs_diffs_1 = np.array(abs_diffs_1)
        abs_diffs_1 = np.nan_to_num(abs_diffs_1)
        is_zero = np.isclose(abs_diffs_1,np.zeros(abs_diffs_1.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.first_mom_weight)
        tst_residual_1 = np.sum(abs_diffs_1) +bonus
        #print (tst_residual_1)
        return tst_residual_1

    # match of descriptor variance values within the current structure only
    @partial(jit, static_argnums=(0))
    def second_moment(self, descriptors):
        #n_atoms=descriptors.shape[0]
        #n_descriptors=descriptors.shape[1]
        vrs = np.var(descriptors,axis=0)
        abs_diffs_2 = np.abs(vrs - descriptors_target_2)
        abs_diffs_2 = np.array(abs_diffs_2)
        abs_diffs_2 = np.nan_to_num(abs_diffs_2)
        is_zero = np.isclose(abs_diffs_2,np.zeros(abs_diffs_2.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.second_mom_weight)
        tst_residual_2 = np.sum(abs_diffs_2) +bonus
        #print (tst_residual_2)
        return tst_residual_2

    #note that the current default weights for this "potential" turn off the variance contribution
    @partial(jit, static_argnums=(0))
    def V(self,descriptors,weights=[1.0,0.0]):
        if self.n_elements > 1:
            descriptors_flt = descriptors.flatten()
        else:
            descriptors_flt = descriptors
        vi = ((weights[0]*self.first_moment(descriptors_flt)) + (weights[1]*self.second_moment(descriptors_flt)))
        vj = ((weights[0]*self.first_moment_cross(descriptors_flt)) + (weights[1]*self.second_moment_cross(descriptors_flt)))
        return self.K_self*vi + self.K_cross*vj

    def __call__(self, elems, descriptors, beta, energy):
        self.last_descriptors=descriptors.copy()
        if self.mode=="run":
            b=descriptors[:,self.mask]
            ener=self.V(b)
            energy[:]=0
            energy[0]=ener
            b=self.V_grad(b)
            if not np.all(np.isfinite(b)):
                print("GRAD ERROR!")
                #print(b)

            beta[:,:]=0
            beta[:,self.mask]=b

        if self.mode=="update":
            b=descriptors[:,self.mask]
            self.update(b)

class GRSSampler:
    def __init__(self, model, before_loading_init):
        self.model=model
        self.lmp = lammps.lammps(cmdargs=['-screen','none'])
        lammps.mliap.activate_mliappy(self.lmp)
        self.lmp.commands_string(before_loading_init)
        lammps.mliap.load_model(model)

    def update_model(self):
        self.model.set_mode_update()
        self.lmp.commands_string("variable etot equal etotal")
        self.lmp.commands_string("variable ptot equal press")
        self.lmp.commands_string("variable pairp equal epair")
        self.lmp.commands_string("variable numat equal atoms")
        self.lmp.commands_string("run 0")
        self.lmp.commands_string("print \"${etot} ${pairp} ${ptot} ${numat} \" append Summary.dat screen no")
        self.model.set_mode_run()

    def run(self,cmd=None):
        if cmd==None:
            self.lmp.commands_string("run 0")
        else:
            self.lmp.commands_string(cmd)

