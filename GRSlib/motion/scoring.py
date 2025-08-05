from GRSlib.parallel_tools import ParallelTools
from GRSlib.motion.lossfunc import moments
import lammps, lammps.mliap
from lammps.mliap.loader import *
from jax import grad, jit
from functools import partial
import numpy as np

#Scoring has to be a class within motion because we want a consistent reference for scores, ans this
#refrence will be LAMMPS using a constructed potential energy surface from the representation loss function

class Scoring:

    def __init__(self, data, current_desc, target_desc, pt, config):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.current_desc = current_desc
        self.target_desc = target_desc
        self.data = data
        self.n_elements = self.config.sections['BASIS'].numtypes
        if self.n_elements > 1:
            current_desc = current_desc.flatten()
            target_desc = target_desc.flatten()
#        print(np.shape(current_desc),np.shape(target_desc))

    @partial(jit, static_argnums=(0))
    def loss_function(self):
        #construct the Jax graph to hand off to lammps
        #TODO This should be generalized to allow for any loss function defined in lossfunc/ to be called
        first_mom = None
        second_mom = None
        third_mom = None
        fourth_mom = None
        set_of_moments = []
        if (any(x == 'mean' for x in self.config.sections['SCORING'].moments)):
            print("Adding mean to loss function force field")
            first_mom = moments.first_moment(self)
        if (any(x == 'stdev' for x in self.config.sections['SCORING'].moments)):
            print("Adding standard deviation to loss function force field")
            second_mom = moments.second_moment(self)
        if (any(x == 'skew' for x in self.config.sections['SCORING'].moments)):
            print("Adding skewness to loss function force field")
            third_mom = moments.third_moment(self)
        if (any(x == 'kurt' for x in self.config.sections['SCORING'].moments)):
            print("Adding kurtosis to loss function force field")
            fourth_mom = moments.fourth_moment(self)
        for item in [second_mom, third_mom, fourth_mom]: 
            if item != None:
                    set_of_moments.append(item)
        loss_ff = first_mom
        for item in set_of_moments:
             loss_ff += item
        return loss_ff

    def construct_lmp(self):
        #Generates the major components of a lammps script needed for a scoring call
        self._lmp = self.pt.initialize_lammps('log.lammps',0)
        lammps.mliap.activate_mliappy(self._lmp)

#        me = self._lmp.extract_setting("world_rank")
#        nprocs = self._lmp.extract_setting("world_size")
#        cmds = ["-screen", "none", "-log", "none"]
#        self._lmp = lammps(cmdargs = cmds)

        self._lmp.command("clear")
        self._lmp.command("units metal")
        self._lmp.command("atom_style atomic")
        self._lmp.command("read_data %s" % self.data)
        #TODO make the possibility to import any reference potential to be used with the mliap one
#        self._lmp.command("pair_style mliap model mliappy LATER descriptor ace coupling_coefficients.yace")
        try:
            soft_strength = self.config.sections['MOTION'].soft_strength
        except:
            soft_strength = 1.0
        #self._lmp.command("pair_style hybrid/overlay soft %2.3f mliap model mliappy LATER descriptor ace coupling_coefficients.yace" % self.config.sections['MOTION'].soft_strength)
        self._lmp.command("pair_style hybrid/overlay soft %2.3f mliap model mliappy LATER descriptor ace coupling_coefficients.yace" % soft_strength)
        #self._lmp.command("pair_coeff * * soft %f" % self.config.sections['MOTION'].soft_strength)
        self._lmp.command("pair_coeff * * soft %f" % soft_strength)
        elements = self.config.sections['BASIS'].elements
        print(elements)
        self._lmp.command("pair_coeff mliap * * %s" % (" ".join(str(x) for x in self.config.sections['BASIS'].elements)))
        self._lmp.command('neighbor  2.3 bin')
        self._lmp.command("neigh_modify one 10000")
        loss_ff = self.loss_function()
        forces = grad(self.loss_function)
        lammps.mliap.load_model(loss_ff)
              
    def get_atomic_energies(self):
        #Return as array per-atom energies for the set of potentials applied
        self.construct_lmp()
        self._lmp.command("compute peatom all pe/atom")
        self._lmp.commands_string("run 0")
        num_atoms = self._lmp.extract_global("natoms")
        atom_energy = _extract_compute_np(self._lmp, "peatom", 0, 2, (num_atoms, 1))
        del self._lmp
        return atom_energy

    def get_norm_forces(self):
        #Return as array per-atom forces 
        self.construct_lmp()
        self._lmp.command("compute fatom all property/atom fx fy fz")
        self._lmp.commands_string("run 0")
        num_atoms = self._lmp.extract_global("natoms")
        atom_forces = _extract_compute_np(self._lmp, "fatom", 0, 2, (num_atoms, 3))        
        del self._lmp
        return atom_forces

    def get_score(self):
        #Return as array unweighted scores per moment
        self.construct_lmp()
        self._lmp.commands_string("run 0")
        score = self._lmp.get_thermo("pe") # potential energy
        del self._lmp
        return score

    def add_cmds_before_score(self,string):
        self.construct_lmp()
        self._lmp.commands_string("run 0")
        before_score = self._lmp.get_thermo("pe") # potential energy
        self._extract_commands(string)
        self._lmp.commands_string("run 0")
        after_score = self._lmp.get_thermo("pe") # potential energy
        del self._lmp
        return before_score, after_score

    def _extract_commands(self,string):
        #Can be given a block of text where it will split them into individual commands
        add_lmp_lines = [x for x in string.splitlines() if x.strip() != '']
        for line in add_lmp_lines:
                self._lmp.command(line)
