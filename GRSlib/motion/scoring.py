from GRSlib.parallel_tools import ParallelTools
from GRSlib.motion.lossfunc.moments import LossFunction
from GRSlib.converters.sections.lammps_base import Base, _extract_compute_np
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
        self.current_desc = []
        self.target_desc = target_desc
        self.data = data
        self.n_elements = self.config.sections['BASIS'].numtypes
        if self.n_elements > 1:
            current_desc = current_desc.flatten()
            target_desc = target_desc.flatten()
        self._lmp = self.pt.initialize_lammps('log.lammps',0)
        lammps.mliap.activate_mliappy(self._lmp)
        self.loss_ff = LossFunction(self.config, self.current_desc, self.target_desc)
    
    def construct_lmp(self):
        #Generates the major components of a lammps script needed for a scoring call
#        me = self._lmp.extract_setting("world_rank")
#        nprocs = self._lmp.extract_setting("world_size")
#        cmds = ["-screen", "none", "-log", "none"]
#        self._lmp = lammps(cmdargs = cmds)
        self.lmp = self.pt.initialize_lammps('log.lammps',0)
        lammps.mliap.activate_mliappy(self.lmp)
        construct_string=\
        """
        units metal
        atom_style atomic
        read_data {}
        pair_style mliap model mliappy LATER descriptor ace coupling_coefficients.yace
        pair_coeff * * {}
        neighbor 2.3 bin
        neigh_modify one 10000
        thermo 10
        thermo_style custom step etotal temp press
        """
        init_lmp=construct_string.format(self.data, (" ".join(str(x) for x in self.config.sections['BASIS'].elements)))
        #TODO make the possibility to import any reference potential to be used with the mliap one
        self.lmp.commands_string(init_lmp)
        lammps.mliap.load_model(self.loss_ff)
        self.lmp.command("run 0")
              
    def get_atomic_energies(self):
        #Return as array per-atom energies for the set of potentials applied
        self.construct_lmp()
        self.lmp.command("compute peatom all pe/atom")
        self.lmp.command("run 0")
        num_atoms = self.lmp.extract_global("natoms")
        atom_energy = _extract_compute_np(self.lmp, "peatom", 0, 2, (num_atoms, 1))
        del self.lmp
        return atom_energy

    def get_norm_forces(self):
        #Return as array per-atom forces 
        self.construct_lmp()
        self._lmp.command("compute fatom all property/atom fx fy fz")
        self._lmp.command("run 0")
        num_atoms = self._lmp.extract_global("natoms")
        atom_forces = _extract_compute_np(self.lmp, "fatom", 0, 2, (num_atoms, 3))        
        del self.lmp
        return atom_forces

    def get_score(self):
        #Return as array unweighted scores per moment
        self.construct_lmp()
        self.lmp.command("run 0")
        score = self.lmp.get_thermo("pe") # potential energy
#        del self._lmp
        return score

    def add_cmds_before_score(self,string):
        self.construct_lmp()
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
