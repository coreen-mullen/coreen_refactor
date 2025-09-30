from GRSlib.Ver0_Files.opt_tools import internal_generate_cell
from GRSlib.parallel_tools import ParallelTools
#from GRSlib.motion.lossfunc.moments import LossFunction
from GRSlib.Ver0_Files.opt_tools import get_desc_count
#from GRSlib.parallel_tools import ParallelTools
#from GRSlib.motion.lossfunc.moments import Moments
#from GRSlib.motion.lossfunc import Gradient
from GRSlib.converters.sections.lammps_base import Base, _extract_compute_np
from examples.simple_test.GRS_protocol import GRSModel, GRSSampler
import lammps, lammps.mliap
from lammps.mliap.loader import *
from functools import partial
import numpy as vnp
from GRSlib.Ver0_Files.opt_tools import *
from GRSlib.GRS import *
#from examples.simple_test.GRS_protocol import *
#Scoring has to be a class within motion because we want a consistent reference for scores, ans this
#refrence will be LAMMPS using a constructed potential energy surface from the representation loss function
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

#Scoring has to be a class within motion because we want a consistent reference for scores, and this
#refrence will be LAMMPS using a constructed potential energy surface from the representation loss function.
#Sub-classes of Scoring will be versions of this representation loss function (Moments, Entropy, etc), allowing
#for custom verions to be added without trouble.

class Scoring:

#    def __init__(self, pt, config, data, loss_ff, **kwargs):
    def __init__(self, pt, config, loss_func, data, descriptors):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.data = data
        self.descriptors = descriptors
        self.loss_func = loss_func
        self.loss_func.__init__(self.pt, self.config, self.descriptors) #Initialize loss function, get ready to send to scoring
        self.loss_func(self.pt, self.config, self.descriptors) #Call loss function, get ready to send to scoring
        self.lmp = self.pt.initialize_lammps('log.lammps',0)
        lammps.mliap.activate_mliappy(self.lmp)

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
        lammps.mliap.load_model(self.loss_func)
        self.lmp.command("run 0")
              
    def get_atomic_energies(self):
        #Return as array per-atom energies for the set of potentials applied
        lammps.mliap.activate_mliappy(self.lmp)
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
    def ensemble_score(self, n_totconfig, data_path, cross_weight, self_weight, randomize_comps, mincellsize, maxcellsize, target_comps, min_typ_global, soft_strength, nelements, n_descs, mask, rand_comp):
        if mask == None:
            mask = range(n_descs)
        self.mask = mask  # Generates the multiple structures
        scores = []  # Initialize a list to store scores
        
        print(f"Starting ensemble_score with {n_totconfig} configurations.")  # Debugging line

        for i in range(1, n_totconfig + 1):
            print(f"Configuration {i}/{n_totconfig} - Using indices: {mask}")  # Debugging line

        # Generate the cell
            g = internal_generate_cell(i, desired_size=vnp.random.choice(range(mincellsize, maxcellsize)), template=None, desired_comps=target_comps, use_template=None, min_typ=min_typ_global, soft_strength=soft_strength)
            print(g)
            print(f"Cell generated for configuration {i}: {g}")  # Debugging line

            em = GRSModel(nelements, n_descs, mask=mask)
            sampler = GRSSampler(em, g)
            em.K_cross = cross_weight
            em.K_self = self_weight

            print(f"Running minimization for configuration {i}.")  # Debugging line
        # Run the minimization process
            sampler.run("minimize 1e-6 1e-6 1000 10000")

            print(f"Minimization completed for configuration {i}. Writing data.")  # Debugging line
        # Write the data to a file
            sampler.run("write_data %s/sample.%i.dat " % (data_path, i))

            print(f"Data written for configuration {i}. Updating model.")  # Debugging line
            sampler.update_model()  # Updating the model combines generated structures

        # Calculate the score for the current configuration
            score = self.get_score(g)  # Pass the generated structure to get_score
            print(f"Score for configuration {i}: {score}")  # Debugging line

            if score is None:
                print(f"Score for configuration {i} is None.")  # Debugging line

            scores.append(score)  # Append the score to the list

        print("Final scores list:", scores)  # Debugging line
        return scores  # Return the list of scores
"""def ensemble_score(self, n_totconfig, data_path, cross_weight, self_weight, randomize_comps, mincellsize, maxcellsize, target_comps, min_typ_global, soft_strength, nelements, n_descs, mask, rand_comp):
        self.mask = mask  # Generates the multiple structures
        scores = []  # Initialize a list to store scores
        print(f"Starting ensemble_score with {n_totconfig} configs.")
        for i in range(1, n_totconfig + 1):
            print(i, "/", n_totconfig, "Using indices:", mask)

            if not randomize_comps:
                g = internal_generate_cell(i, desired_size=vnp.random.choice(range(mincellsize, maxcellsize)), template=None, desired_comps=target_comps, use_template=None, min_typ=min_typ_global, soft_strength=soft_strength)
            else:
                target_comps_rnd = rand_comp(target_comps)  # Randomize compositions
                g = internal_generate_cell(i, desired_size=vnp.random.choice(range(mincellsize, maxcellsize)), template=None, desired_comps=target_comps_rnd, use_template=None, min_typ=min_typ_global, soft_strength=soft_strength)

            em = GRSModel(nelements, n_descs, mask=mask)
            sampler = GRSSampler(em, g)
            em.K_cross = cross_weight
            em.K_self = self_weight

        # Run the minimization process
            sampler.run("minimize 1e-6 1e-6 1000 10000")

        # Write the data to a file
            sampler.run("write_data %s/sample.%i.dat " % (data_path, i))

            sampler.update_model()  # Updating the model combines generated structures

        # Calculate the score for the current configuration
            score = self.get_score()  # Assuming get_score is defined to return the score for the current state
            if score is None:
                print(f"Score for configuration {i} is None.")
            scores.append(score)  # Append the score to the list
        print("Final scores list:", scores)
        return scores  # Return the list of scores
"""
"""    def ensemble_score(self, n_totconfig, data_path,cross_weight, self_weight, randomize_comps, mincellsize, maxcellsize, target_comps, min_typ_global,soft_strength, nelements,n_descs,mask,rand_comp): 
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
    
"""
# total number of configurations
#n_totconfig = int(sys.argv[2])
               # self.lmp.command(line)
