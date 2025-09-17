from GRSlib.Ver0_Files.opt_tools import internal_generate_cell
from GRSlib.parallel_tools import ParallelTools
from GRSlib.motion.lossfunc.moments import LossFunction
from GRSlib.Ver0_Files.opt_tools import get_desc_count
from GRSlib.converters.sections.lammps_base import Base, _extract_compute_np
from examples.simple_test.GRS_protocol import GRSModel, GRSSampler
import lammps, lammps.mliap
from lammps.mliap.loader import *
from jax import grad, jit
from functools import partial
import numpy as vnp
from opt_tools.py import *
from GSQS_protocol import *
#Scoring has to be a class within motion because we want a consistent reference for scores, ans this
#refrence will be LAMMPS using a constructed potential energy surface from the representation loss function
n_totconfig = 10
data_path = 'bcc.data'
cross_weight =1.000000
self_weight: 1.000000
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
class Scoring:

    def __init__(self, data, current_desc, target_desc, prior_desc, pt, config):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.current_desc = []
        self.target_desc = target_desc
        self.prior_desc = prior_desc
        self.data = data
        self.n_elements = self.config.sections['BASIS'].numtypes
        if self.n_elements > 1:
            current_desc = current_desc.flatten()
            target_desc = target_desc.flatten()
        self.lmp = self.pt.initialize_lammps('log.lammps',0)
        lammps.mliap.activate_mliappy(self.lmp)
        self.loss_ff = LossFunction(self.config, self.current_desc, self.target_desc, self.prior_desc)
    
    def construct_lmp(self):
        #Generates the major components of a lammps script needed for a scoring call
#        me = self.lmp.extract_setting("world_rank")
#        nprocs = self.lmp.extract_setting("world_size")
#        cmds = ["-screen", "none", "-log", "none"]
#        self.lmp = lammps(cmdargs = cmds)
        self.lmp = self.pt.initialize_lammps('log.lammps',0)
        lammps.mliap.activate_mliappy(self.lmp)
        construct_string=\
        """
        units metal
        atom_style atomic
        read_data {}
        pair_style hybrid/overlay soft 1.0 mliap model mliappy LATER descriptor ace coupling_coefficients.yace
        pair_coeff * * soft {}
        pair_coeff * * mliap {}
        neighbor 2.3 bin
        neigh_modify one 10000
        thermo 10
        thermo_style custom step etotal temp press
        """
        init_lmp=construct_string.format(self.data, self.config.sections["MOTION"].soft_strength, (" ".join(str(x) for x in self.config.sections['BASIS'].elements)))
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
#        del self.lmp
        return atom_energy

    def get_norm_forces(self):
        #Return as array per-atom forces 
        self.construct_lmp()
        self.lmp.command("compute fatom all property/atom fx fy fz")
        self.lmp.command("run 0")
        num_atoms = self.lmp.extract_global("natoms")
        atom_forces = _extract_compute_np(self.lmp, "fatom", 0, 2, (num_atoms, 3))        
#        del self.lmp
        return atom_forces

    def get_score(self):
        #Return as array unweighted scores per moment
        self.construct_lmp()
        self.lmp.command("run 0")
        score = self.lmp.get_thermo("pe") # potential energy
#        del self.lmp
        return score

    def add_cmds_before_score(self,string):
        self.construct_lmp()
        before_score = self.get_score()
        self._extract_commands(string)
        self.lmp.commands_string("run 0")
        after_score = self.lmp.get_thermo("pe") # potential energy
#        del self._lmp
        return before_score, after_score

    def _extract_commands(self,string):
        #Can be given a block of text where it will split them into individual commands
        add_lmp_lines = [x for x in string.splitlines() if x.strip() != '']
        for line in add_lmp_lines:
            self.lmp.command(line)
#internal generate cell is only used in scoring.py but defined in opt_tools + gsqsmodel only in scoring.py currently
#class ensemble_score(): #will take in the target, and compare it to multiple generated structures -- look at fitsnap sections
# min and maxatoms the same as maxcellsize?
# target_comps is in input
#numelements = num types? 

    def ensemble_score(self, n_totconfig, data_path,cross_weight, self_weight, randomize_comps, mincellsize, maxcellsize, target_comps, min_typ_global,soft_strength, nelements,n_descs,mask,rand_comp): 
        self.mask=mask#generates the multiple structures -- needs internal generate cell , some of these should be defined in the input file like n_totconfig if they choose multiple and the crossweight and self weigths
        i = 1
        while i <= n_totconfig: 
            print(i,"/",n_totconfig,"Using indicies :",mask)
        if not randomize_comps:
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=None,desired_comps=target_comps,use_template=None,min_typ=min_typ_global,soft_strength=soft_strength)
        else:
            target_comps_rnd = rand_comp(target_comps) #randomize_comp in input? and target_comps in input?
            g = internal_generate_cell(i,desired_size=vnp.random.choice(range(mincellsize,maxcellsize)),template=None,desired_comps=target_comps_rnd,use_template=None,min_typ=min_typ_global,soft_strength=soft_strength)
        em=GRSModel(nelements,n_descs,mask=mask) 
        sampler = GRSSampler(em, g)
        em.K_cross = cross_weight
        em.K_self = self_weight
        #min type
        sampler.run("minimize 1e-6 1e-6 1000 10000")
        
        sampler.run("write_data %s/sample.%i.dat " % (data_path, i))
        
        sampler.update_model() #updating the model is how all of the generated structures get combined into one.
    

# total number of configurations
#n_totconfig = int(sys.argv[2])
               # self.lmp.command(line)
