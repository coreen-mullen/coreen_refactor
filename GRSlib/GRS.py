from GRSlib.parallel_tools import ParallelTools
from GRSlib.io.input import Config
from GRSlib.converters.convert_factory import convert
from GRSlib.motion.scoring import Scoring
from GRSlib.motion.motion import Gradient, Genetic

import random
import numpy as np
import copy

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
        self.target_desc = []
        self.current_desc = []

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
        print("Called Convert To Descriptors for %s" % data)
        self.convert = convert(self.config.sections['BASIS'].descriptor,self.pt,self.config) 
        descriptors = self.convert.run_lammps_single(data)
            
        return descriptors

    def get_score(self,data):
        """
        Accepts a structure (xyz) as input and will return descriptors (D), optionally will convert
        between file types (xyz=lammps-data, ase.Atoms, etc)
        """
        #Pass data to, and do something with the functs of scoring
#        target_desc = self.convert_to_desc(self.config.sections['TARGET'].target_fname)
        if not self.target_desc:
            print('Target structure descriptors not found, using starting target')
            self.target_desc = self.convert_to_desc(self.config.sections['TARGET'].target_fname)
        
        self.current_desc = self.convert_to_desc(data)

        if (np.shape(self.current_desc[1])==np.shape(self.target_desc[1])):
            print("Called Scoring Function")
            self.score = Scoring(data, self.current_desc, self.target_desc, self.pt, self.config) 
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
            #1)
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

        if data == None:
            data = self.propose_structure()

        self.current_desc = self.convert_to_desc(data) 
        self.genmove = Genetic(data, self.current_desc, self.target_desc, self.pt, self.config) 
        #Dont want to make a func call the default here since the user will define this?
        #Need a fallback to provide a good default if a genetic move is called.
        #self.genmove.tournament_selection()

    def gradient_move(self,data):
        """
        Accepts a structure (xyz, ase.Atoms) as input and will return updated structure (xyz, ase.Atoms) that 
        has been modified by motion of atoms on the loss function potential
        """
#        @self.pt.single_timeit 
#        def gradient_move():
#        gradient_move()
        #1) Take in target descriptors, convert to moments of descriptor distribution
        #2) Take in current descriptors, convert to moments of descriptor distribution
        #3) Construct a fictitious potential energy surface based on difference in moments
        #4) Assemble a LAMMPS input script that overlaps potentials and runs dynamics
        #5) Return an updated structure and scoring on the difference in moments
        print("Called Gradient_Move")
        
        if data == None:
            data = self.propose_structure()
        
        self.current_desc = self.convert_to_desc(data)
        self.gradmove = Gradient(data, self.current_desc, self.target_desc, self.pt, self.config) 
        if self.config.sections['MOTION'].min_type == 'fire':
            before_score, after_score = self.gradmove.fire_min()
            print(before_score, after_score)
        elif self.config.sections['MOTION'].min_type == 'line':
            before_score, after_score = self.gradmove.line_min()
            print(before_score, after_score)
        elif self.config.sections['MOTION'].min_type == 'box':
            before_score, after_score = self.gradmove.box_min()
            print(before_score, after_score)
        elif self.config.sections['MOTION'].min_type == 'temp':
            before_score, after_score = self.gradmove.run_then_min()
            print(before_score, after_score)

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
        @self.pt.single_timeit
        def write_output():
            print("Doing some output now")
            #self.output.write_lammps(self.solver.fit)
            #self.output.write_errors(self.solver.errors)
        write_output()
