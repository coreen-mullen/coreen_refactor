import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, random
from functools import partial
import lammps, lammps.mliap
from lammps.mliap.loader import *
import copy

class LossFunction:
    def __init__(self, config, current_desc, target_desc):
        self.n_params = 1
        self.config = config
        self.current_desc = current_desc.copy()
        self.target_desc = target_desc.copy() #copy.deepcopy(target_desc)
        self.loss_ff_grad = grad(self.construct_loss)
        self.n_descriptors = np.shape(self.target_desc)[1]
        self.n_elements = self.config.sections['BASIS'].numtypes
        self.grad_first_moment_atomic = grad(self.first_moment_peratom)
        self.grad_first_moment = grad(self.first_moment)
        self.grad_second_moment = grad(self.second_moment)
        self.grad_third_moment = grad(self.third_moment)
        self.grad_fourth_moment = grad(self.fourth_moment)
        self.grad_loss = grad(self.construct_loss)
        self.mask = list(range(self.n_descriptors))
        self.mode="update"

    def __call__(self, elems, current_desc, beta, energy): 
        #The arguments that this function brings in are super improtant and are expected by LAMMPS MLIAP package.
        #LAMMPS will populate the descriptors as a per-atom array into current_desc.
        #Per-atom forces are expected for beta
        #Per-atom energy is expected for energy, need to do some testing if per-atom values can be reported back.
        self.n_atoms = np.shape(current_desc)[0]
        if self.mode=="score":     
            score = self.construct_loss(current_desc, self.target_desc)
            energy[:] = 0
            energy[0] = score
            #print("     Mean score calculated from loss function constructor: ",score)
            forces = self.grad_loss(current_desc, self.target_desc)
            #forces = self.grad_first_moment_atomic(self.current_desc, self.target_desc)
            beta[:,:] = 0
            beta[:,self.mask] = forces
        elif self.mode=="update":
            self.update()
            beta = self.grad_loss(current_desc, self.target_desc)


    def set_mode_update(self):
        self.mode="update"

    def set_mode_score(self):
        self.mode="score"

    def update(self):
        #print("Updating the loss function given new information")
        if self.n_elements > 1:
            self.current_desc = self.current_desc.flatten()
            self.target_desc = self.target_desc.flatten()
        self.mode = "score"

    @partial(jit, static_argnums=(0,))
    def construct_loss(self, current_desc, target_desc):
        #This needs to be a dynamic call like is done in descriptor calcs 
        loss_ff = 0
        set_of_moments = []

        if (any(x == 'mean' for x in self.config.sections['SCORING'].moments)):
            print("Adding mean to loss function force field")
            first_mom = self.first_moment(current_desc, target_desc)
        else:
            first_mom = None
            
        if (any(x == 'stdev' for x in self.config.sections['SCORING'].moments)):
            print("Adding standard deviation to loss function force field")
            second_mom = self.second_moment(current_desc, target_desc)
        else:
            second_mom = None

        if (any(x == 'skew' for x in self.config.sections['SCORING'].moments)):
            print("Adding skewness to loss function force field")
            third_mom = self.third_moment(current_desc, target_desc)
        else:
            third_mom = None

        if (any(x == 'kurt' for x in self.config.sections['SCORING'].moments)):
            print("Adding kurtosis to loss function force field")
            fourth_mom = self.fourth_moment(current_desc, target_desc)
        else:
            fourth_mom = None

        for item in [first_mom, second_mom, third_mom, fourth_mom]: 
            if item != None:
                    set_of_moments.append(item)

        for item in set_of_moments:
             loss_ff += item
        return loss_ff

    @partial(jit, static_argnums=(0,))
    def first_moment(self, current_desc, target_desc):
        current_avg = jnp.average(current_desc, axis=0)
        target_avg = jnp.average(target_desc, axis=0)
        tst_residual = jnp.sum(jnp.nan_to_num(jnp.abs(current_avg-target_avg)))
        is_zero = jnp.array(jnp.isclose(tst_residual,jnp.zeros(tst_residual.shape)),dtype=int)
        bonus = -jnp.sum(is_zero*(float(self.config.sections['SCORING'].moment_bonus[0])))
        tst_residual_final = tst_residual*float(self.config.sections['SCORING'].moments_coeff[0]) + bonus #MAE + bonus
        return tst_residual_final

    @partial(jit, static_argnums=(0,))
    def second_moment(self, current_desc, target_desc):
        current_std = jnp.std(current_desc, axis=0)
        target_std = jnp.std(target_desc, axis=0)
        tst_residual = jnp.sum(jnp.nan_to_num(jnp.abs(current_std-target_std)))
        is_zero = jnp.array(jnp.isclose(tst_residual,jnp.zeros(tst_residual.shape)),dtype=int)
        bonus=-jnp.sum(is_zero*float(self.config.sections['SCORING'].moment_bonus[1]))
        tst_residual_final = tst_residual*float(self.config.sections['SCORING'].moments_coeff[1]) + bonus #MAE + bonus
        return tst_residual_final

    @partial(jit, static_argnums=(0,))
    def third_moment(self, current_desc, target_desc):
        #Showing my work for Pearsons skew = (3(mean-median)/stdev))
        current_avg = jnp.average(current_desc, axis=0)
        target_avg = jnp.average(target_desc, axis=0)
        current_std = jnp.std(current_desc, axis=0)
        target_std = jnp.std(target_desc, axis=0)
        current_med = jnp.median(current_desc, axis=0)
        target_med = jnp.median(target_desc, axis=0)

        current_skew = 3.0*(current_avg-current_med)/current_std
        target_skew = 3.0*(target_avg-target_med)/target_std

        tst_residual = jnp.sum(jnp.nan_to_num(jnp.abs(current_skew-target_skew)))
        is_zero = jnp.array(jnp.isclose(tst_residual,jnp.zeros(tst_residual.shape)),dtype=int)
        bonus=-jnp.sum(is_zero*float(self.config.sections['SCORING'].moment_bonus[2]))
        tst_residual_final = tst_residual*float(self.config.sections['SCORING'].moments_coeff[2]) + bonus #MAE + bonus
        return tst_residual_final

    @partial(jit, static_argnums=(0,))
    def fourth_moment(self, current_desc, target_desc):
        #Showing my work for Kurtosis = Avg(z^4.0)-3 where z=(x-avg(x))/stdev(x)
        current_avg = jnp.average(current_desc, axis=0)
        target_avg = jnp.average(target_desc, axis=0)
        current_std = jnp.std(current_desc, axis=0)
        target_std = jnp.std(target_desc, axis=0)

        current_kurt = jnp.average(((current_desc-current_avg)/current_std)**4.0)-3.0 
        target_kurt = jnp.average(((target_desc-target_avg)/target_std)**4.0)-3.0 

        tst_residual = jnp.sum(jnp.nan_to_num(jnp.abs(current_kurt-target_kurt)))
        is_zero = jnp.array(jnp.isclose(tst_residual,jnp.zeros(tst_residual.shape)),dtype=int)
        bonus=-jnp.sum(is_zero*float(self.config.sections['SCORING'].moment_bonus[3]))
        tst_residual_final = tst_residual*float(self.config.sections['SCORING'].moments_coeff[3]) + bonus #MAE + bonus
        return tst_residual_final

    @partial(jit, static_argnums=(0,))
    def first_moment_peratom(self, current_desc, target_desc):
        target_avg = jnp.average(target_desc, axis=0)
        tst_residual = jnp.nan_to_num(jnp.abs(current_desc-target_avg))
        return tst_residual
