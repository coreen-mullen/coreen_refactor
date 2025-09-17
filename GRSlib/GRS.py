from GRSlib.parallel_tools import ParallelTools
from GRSlib.io.input import Config
from GRSlib.converters.convert_factory import convert
from GRSlib.motion.scoring import Scoring
from GRSlib.motion.motion import Gradient, Genetic

import random, copy, os, glob, shutil
import numpy as np

class GRS:
    """ 
    >Big comment goes here explaining what the code does
    Args:
        input (str): Optional dictionary or path to input file when using library mode; defaults to 
                     None for executable use.
        comm: Optional MPI communicator when using library mode; defaults to None.

    Attributes:
        pt (:obj:`class` ParallelTools): Instance of the ParallelTools class for helping MPI 
                                         communication and shared arrays.
        config (:obj:`class` Config): Instance of the Config class for initializing settings, 
                                      initialized with a ParallelTools instance.
        >Update once more of the code structure is fleshed out
    """
    def __init__(self, input=None, comm=None, arglist: list=[]):
        self.comm = comm
        # Instantiate ParallelTools and Config instances belonging to this GRS instance.
        # NOTE: Each proc in `comm` creates a different `pt` object, but shared arrays still share 
        #       memory within `comm`.
        self.pt = ParallelTools(comm=comm)
        self.pt.all_barrier()
        self.config = Config(self.pt, input, arguments_lst=arglist)
        self.convert = convert(self.config.sections['BASIS'].descriptor,self.pt,self.config) 
        self.target_desc = []
        self.current_desc = []
        self.prior_desc = []

#       Instantiate other backbone attributes.
#       self.basis = basis(self.config.sections["BASIS"].descriptor, self.pt, self.config) if "BASIS" in self.config.sections else None

        # Check LAMMPS version if using nonlinear solvers.
        if (hasattr(self.pt, "lammps_version")):
            if (self.pt.lammps_version < 20220915):
                raise Exception(f"Please upgrade LAMMPS to 2022-09-15 or later to use MLIAP based structure searching.")

    def __del__(self):
        """Override deletion statement to free shared arrays owned by this instance."""
        self.pt.free()
        del self

    def __setattr__(self, name: str, value):
        """
        Override set attribute statement to prevent overwriting important attributes of an instance.
        """
        protected = ("pt", "config")
        if name in protected and hasattr(self, name):
            raise AttributeError(f"Overwriting {name} is not allowed; instead change {name} in place.")
        else:
            super().__setattr__(name, value)

    def convert_to_desc(self,data):
        """
        Accepts a structure (xyz) as input and will return descriptors (D), optionally will convert
        between file types (xyz=lammps-data, ase.Atoms, etc)
        """
        #Pass data to, and do something with the functs of convert
        #print("Called Convert To Descriptors for %s" % data)
        descriptors = self.convert.run_lammps_single(data)
#        if self.config.sections['OUTPUT'].verbosity:
        if not os.path.exists('%s.npy'%data):
            np.save('%s.npy'%data.removesuffix(".data"), descriptors)
            
        return descriptors

    def set_prior(self,structs):
        """
        Accepts a structure (xyz) as input and will return descriptors (D) to be stored and used in conjunction with
        the target, optionally will convert between file types (xyz=lammps-data, ase.Atoms, etc). This is available 
        whena single structure cannot reproduce the target.
        """
        for data in structs:
            prior_desc = self.convert.run_lammps_single(data)
            if len(self.prior_desc)==0:
                self.prior_desc = prior_desc
            else:
                self.prior_desc = np.r_[self.prior_desc, prior_desc]
        np.save('prior.npy', self.prior_desc)
        return self.prior_desc

    def update_start(self,data,str_option):
        """
        Used to update the starting structure, usually after a genetic/gradient move has been applied.
        """
        #Save the last structure in a meaningful way, check if other data is present already
        store_id = len(glob.glob(self.config.sections['TARGET'].job_prefix+"*.data"))
        if store_id==0:
            data = self.config.sections['TARGET'].start_fname
        shutil.copyfile(data, self.config.sections['TARGET'].job_prefix + "_%s.data"%store_id)

        if str_option == "Continue":
            #Take the last state from either genetic/gradient move and continue with next move
            data = self.config.sections['TARGET'].job_prefix + "_%s.data"%store_id
        elif str_option == "MinScore":
            #If the score has decreased, continue. Else, retry last structure.
            if(self.before_score >= self.after_score):
                data = self.config.sections['TARGET'].job_prefix + "_%s.data"%store_id
            else:
                data = self.config.sections['TARGET'].job_prefix + "_%s.data"%(store_id-1)
        elif str_option == "MaxScore":
            #If the score has increased, continue. Else, retry last structure.
            if(self.before_score < self.after_score):
                data = self.config.sections['TARGET'].job_prefix + "_%s.data"%store_id
            else:
                data = self.config.sections['TARGET'].job_prefix + "_%s.data"%(store_id-1)
#        elif str_option == "Template":
#            #Call structure builder from template, see James' old code
#        elif str_option == "Random":
#            #Call structure buider that creates a random cell of atoms
        elif str_option == "Reset":
            #Fallback to the original input strcture
            data = self.config.sections['TARGET'].start_fname
        elif str_option == "Last":
            #Fallback to the last structure before the most recent genetic/gradient move.
            data = self.config.sections['TARGET'].job_prefix + "_%s.data"%(store_id-1)
        else:
            print("You did not specify a continuation condition (Continue, MinScore, MaxScore, Template, Random, Reset, Last), exiting")
            exit()
        return data

    def get_score(self,data):
        """
        Accepts a structure (xyz) as input and will return descriptors (D), optionally will convert
        between file types (xyz=lammps-data, ase.Atoms, etc)
        """
        #Pass data to, and do something with the functs of scoring
        if self.config.sections['TARGET'].target_fname == None:
            print("Provided target descriptors superceed target data file")
            self.target_desc = np.load(self.config.sections['TARGET'].target_fdesc)    
        else:
            self.target_desc = self.convert_to_desc(self.config.sections['TARGET'].target_fname)
        self.current_desc = self.convert_to_desc(data)
        
        if self.prior_desc == None: 
            self.prior_desc = self.set_prior(self.config.sections['TARGET'].start_fname)

        if (np.shape(self.current_desc[1])==np.shape(self.target_desc[1])):
            self.score = Scoring(data, self.current_desc, self.target_desc, self.prior_desc, self.pt, self.config) 
            score = self.score.get_score()
        else:
            raise RuntimeError(">>> Found unmatched BASIS for target and current descriptors")
            
        return score

    def propose_structure(self):
        """
        Propose new structure from random, ase, or templates.
        """
        @self.pt.single_timeit
        def propose_structure():
            print("Called Propose_Structure")
        propose_structure()

    def genetic_move(self,data):
        """
        Hybridize or mutate a structure using a set of moves sampled via a genetic algorithm
        """
#        @self.pt.single_timeit
#        def genetic_move():
#        genetic_move()
        #1) Propose a set of structures from templates, ase, or random (or read in a list of ase.Aatoms objects)
        #2) Score each of the candidates (ase_to_lammps -> run_single)
        #3) Hybridize, Mutate based on set of rules and probabilities
        #4) Store socring information with best-of-generation and best-overall isolated
        #5) Loop until generation limit or scoring residual below threshold
        print("Called Genetic_Move")
        if self.current_desc is None:
            self.current_desc = self.propose_structure()
        #if data == None:
            #data = self.propose_structure()

        self.current_desc = self.convert_to_desc(data)
        self.target_desc = self.convert_to_desc(self.config.sections['TARGET'].target_fname) 
        self.genmove = Genetic(data, self.current_desc, self.target_desc, self.pt, self.config) 
        #Dont want to make a func call the default here since the user will define this?
        #Need a fallback to provide a good default if a genetic move is called.
        #self.genmove.tournament_selection()

#    @self.pt.single_timeit 
    def gradient_move(self,data):
        """
        Accepts a structure (xyz, ase.Atoms) as input and will return updated structure (xyz, ase.Atoms) that 
        has been modified by motion of atoms on the loss function potential
        """
#        @self.pt.single_timeit 
#        def gradient_move():
#        gradient_move()
        #1) Take in target descriptors, convert to moments of descriptor distribution or other loss function
        #2) Take in current descriptors, convert to moments of descriptor distribution or other loss function
        #3) Construct a fictitious potential energy surface based on difference in moments or other loss function
        #4) Assemble a LAMMPS input script that overlaps potentials and runs dynamics
        #5) Return an updated structure and scoring on the difference in moments or other loss function
#        print("Called Gradient_Move")
        
        if data == None:
            data = self.propose_structure()
        self.current_desc = self.convert_to_desc(data)
        if self.config.sections['TARGET'].target_fname == None:
            print("Provided target descriptors superceed target data file")
            self.target_desc = np.load(self.config.sections['TARGET'].target_fdesc)    
        else:
            self.target_desc = self.convert_to_desc(self.config.sections['TARGET'].target_fname)
   
        self.gradmove = Gradient(data, self.current_desc, self.target_desc, self.prior_desc, self.pt, self.config) 
        if self.config.sections['MOTION'].min_type == 'fire':
            self.before_score, self.after_score, data = self.gradmove.fire_min()
        elif self.config.sections['MOTION'].min_type == 'line':
            self.before_score, self.after_score, data = self.gradmove.line_min()
        elif self.config.sections['MOTION'].min_type == 'box':
            self.before_score, self.after_score, data = self.gradmove.box_min()
        elif self.config.sections['MOTION'].min_type == 'temp':
            self.before_score, self.after_score, data = self.gradmove.run_then_min()
        
        self.write_output()
        
        return data

    def baseline_training(self):
        """
        Accepts a structure (xyz, ase.Atoms) as input and will return updated structure (xyz, ase.Atoms) that 
        has been modified by motion of atoms on the loss function potential
        """
        @self.pt.single_timeit
        def baseline_training():
            #This is more of a 'super' function because it will call many other routines to give the result of a
            #baseline training set with many (hundreds? thousands?) candidate structures. Should return a score
            #of the training diversity based on moments of the descriptor distribution.
            print("Called Baseline_Training")
        baseline_training()

    def write_output(self):  
        #Can modfiy to take in an output style and call a specific function, for now it is generic and redundant naming
#        @self.pt.single_timeit
        def text_output():
            store_id = len(glob.glob(self.config.sections['TARGET'].job_prefix+"*"))
            with open("scoring_%s.txt"%self.config.sections['TARGET'].job_prefix, "a") as f:
                print(store_id,self.before_score, self.after_score, file=f)
        text_output()
