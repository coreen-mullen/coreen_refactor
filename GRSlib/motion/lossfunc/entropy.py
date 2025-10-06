from GRSlib.motion.scoring import Scoring
import jax.numpy as jnp
import numpy as np
import scipy as sp
from jax import grad, jit
from functools import partial

class Entropy(Scoring):
    def __init__(self, *args): #pt, config, target_desc, prior_desc):
        self.pt, self.config, descriptors = args
        self.target_desc = descriptors.get('target',None).copy() 
        self.prior_desc = descriptors.get('prior',None).copy()
        self.n_descriptors = np.shape(self.target_desc)[1]
        self.mask = list(range(self.n_descriptors))
        self.cholesky_decomp = np.identity(len(self.mask))
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

            # Energy and forces where prior descriptors are used to define unit transformation
            self.transform_desc(self.prior_desc)
            score = self.construct_loss(current_desc)
            energy[:] = 0
            energy[0] = score #*float(self.config.sections["SCORING"].strength_prior) 
            forces = self.grad_loss(current_desc) 
            beta[:,:]= 0
            beta[:,self.mask] = forces*float(self.config.sections["SCORING"].strength_prior) 

            # Energy and forces where target descriptors are used to define unit transformation
            self.transform_desc(self.prior_desc)
            score = self.construct_loss(current_desc)
            energy[0] += score #*float(self.config.sections["SCORING"].strength_target)
            forces = self.grad_loss(current_desc)
            beta[:,self.mask] += forces*float(self.config.sections["SCORING"].strength_target)

        elif self.mode=="update":
            self.update(args)
            beta = self.grad_loss(self.target_desc)


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
    def construct_loss(self, current_desc):
        loss_ff = float(self.config.sections["SCORING"].internal_entropy)*self.internal_entropy(current_desc) + float(self.config.sections["SCORING"].ensemble_entropy)*self.ensemble_entropy(current_desc)
        return loss_ff

#    @partial(jit, static_argnums=(0,))
    def transform_desc(self,input_desc):
        """
        Transform (Cholesky Decomposition) the input descriptors into a unit vector basis, used to project other descriptors into this unit space
        """
        sum_desc = jnp.sum(input_desc,axis=0)
        sum_products = input_desc.T@input_desc
        covariance = sum_products/self.n_descriptors - jnp.outer(sum_desc/self.n_descriptors, sum_desc/self.n_descriptors)
        self.cholesky_decomp = jnp.linalg.cholesky(jnp.linalg.inv(covariance))
        #No return needed, Cholesky decompositon is all that is needed to be re-computed

    @partial(jit, static_argnums=(0,))
    def internal_entropy(self, current_desc):
        """
        Computes information entropy of current structure relative to prior descriptors
        """
        internal_score = 0.0
        atoms_curr = np.shape(current_desc)[0]
        transformed_current = (self.cholesky_decomp@current_desc.T).T
        silverman_width = jnp.identity(self.n_descriptors)
        silverman_width *= ((atoms_curr**(-1./(self.n_descriptors+4.)))*(4./(self.n_descriptors+2.)**(1./(self.n_descriptors+4.)) ))**2
        covariance_current = ((jnp.linalg.inv(silverman_width))@(transformed_current).T).T
        for i in range(atoms_curr):
            rho = self.density(i,transformed_current,covariance_current,atoms_curr)
            internal_score += -jnp.log(rho)
        internal_score /= atoms_curr
        return internal_score

    @partial(jit, static_argnums=(0,))
    def ensemble_entropy(self, current_desc):
        """
        Computes per-atom scores in the current structure relative to target descriptors
        """
        ensemble_score = 0.0
        atoms_curr = np.shape(current_desc)[0]

        for i in range(atoms_curr):
            atomic_desc = self.cholesky_decomp@((current_desc[i]- jnp.average(self.target_desc,axis=0)))
            ensemble_score += 0.5*(atomic_desc@atomic_desc)
        ensemble_score /= atoms_curr
        return ensemble_score

    @partial(jit, static_argnums=(0,1))
    def density(self,i,wd,wdh,atoms_curr):
        wdr=wd[i,None]-wd
        wdhr=wdh[i,None]-wdh
        rho=jnp.sum(jnp.exp(-0.5*jnp.sum(wdr*wdhr,axis=1)))/atoms_curr
        return rho
