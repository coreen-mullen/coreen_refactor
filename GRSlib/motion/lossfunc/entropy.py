from GRSlib.motion.scoring import Scoring
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, random
from functools import partial
import lammps, lammps.mliap
from lammps.mliap.loader import *
import copy

class Entropy(Scoring):
    def __init__(self, *args): #pt, config, target_desc, prior_desc):
        self.pt, self.config, descriptors = args
        self.target_desc = descriptors.get('target',None).copy() 
        self.prior_desc = descriptors.get('prior',None).copy()
        self.n_descriptors = np.shape(self.target_desc)[1]
        self.mask = list(range(self.n_descriptors))
        self.n_params = 1 #Variables LAMMPS needs to know about
        self.n_elements = self.config.sections['BASIS'].numtypes #Variables LAMMPS needs to know about
        self.loss_ff_grad = grad(self.construct_loss)
        self.grad_loss = grad(self.construct_loss)
        self.mode = "update"

    def __call__(self, *args):
        #The arguments that this function brings in are super improtant and are expected by LAMMPS MLIAP package.
        #They are (elems, current_desc, beta, energy)
        #Integer values of LAMMPS atom types are in elems
        #Descriptors as a per-atom array into current_desc.
        #Per-atom forces are expected for beta
        #Per-atom energy is expected for energy, need to do some testing if per-atom values can be reported back.
        if self.mode=="score":     
            elems, current_desc, beta, energy = args
            self.n_atoms = np.shape(current_desc)[0]
            score = self.construct_loss(current_desc, self.target_desc)
            energy[:] = 0
            energy[0] = float(self.config.sections["SCORING"].strength_target)*score #Scaled score (energy) between current and target
            forces = self.grad_loss(current_desc, self.target_desc) #Forces between current and target
            beta[:,:]= 0
            beta[:,self.mask] = float(self.config.sections["SCORING"].strength_target)*forces #Scaled forces between current and target

            score = self.construct_loss(current_desc, self.prior_desc)
#            energy[0] += float(self.config.sections["SCORING"].strength_prior)*score #Scaled score (energy) between current and prior
#            print("     Target, Prior Scores: ", energy[0], score)
            forces = self.grad_loss(current_desc, self.prior_desc) #Forces between current and prior structures
            beta[:,self.mask] += float(self.config.sections["SCORING"].strength_prior)*forces #Scaled forces between current and prior

        elif self.mode=="update":
            self.update(args)
            beta = self.grad_loss(self.target_desc, self.target_desc)


    def set_mode_update(self):
        self.mode="update"

    def set_mode_score(self):
        self.mode="score"

    def update(self,*args):
        pt, config, descriptors = args[0]
        self.target_desc = descriptors.get('target',None).copy()
        self.prior_desc = descriptors.get('prior',None).copy()
        self.n_descriptors = np.shape(self.target_desc)[1]
        self.mask = list(range(self.n_descriptors))

        if self.n_elements > 1:
            self.current_desc = self.current_desc.flatten()
            self.target_desc = self.target_desc.flatten()
            self.prior_desc = self.prior_desc.flatten()
        self.mode = "score"

    @partial(jit, static_argnums=(0,))
    def construct_loss(self, current_desc, target_desc):
        #This needs to be a dynamic call like is done in descriptor calcs 
        loss_ff = 0

        return loss_ff

    @partial(jit, static_argnums=(0,))
    def calc(self, current_desc, target_desc):
        """
        Something done here
        """
        return tst_residual_final
