#from GRSlib.parallel_tools import ParallelTools
#from GRSlib.motion.scoring_factory import scoring
from GRSlib.motion.scoring import Scoring
from GRSlib.motion.genetic import Genetic
from GRSlib.motion.create import Create
import numpy as np
import random

# Two types of motion (aka changes) can be applied to a structure 1) Gradients of the loss function (energy/score) 
# yielding continuous changes, or 2) discrete moves that include (atom addition/removal, chemical identities).
# New types can be added as classes if there is need, follow class inheritence of existing motion types.
# These classes require a scoring function, which is inherited after initializing in GRS.py. 

# A key difference between Gradient and Optimize is that the former operates on a single structure at a time, while
# the latter will create a dictionary of structures and their scores to be evaluated. 

class Gradient:

    def __init__(self, pt, config, data, scoring):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.data = data
        self.scoring = scoring

    def fire_min(self):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """min_style  fire
        min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 100000 halfstepback no initialdelay no
        dump 1 all custom 1 minimize_fire.dump id type x y z fx fy fz
        displace_atoms all random 0.1 0.1 0.1 %s units box
        minimize 1e-6 1e-6 %s %s
        write_data %s_last.data""" % (np.random.randint(low=1, high=99999),self.config.sections['MOTION'].nsteps, self.config.sections['MOTION'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

    def line_min(self):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """min_style  cg
        min_modify dmax 0.05 line quadratic
        dump 1 all custom 1 minimize_line.dump id type x y z fx fy fz
        displace_atoms all random 0.1 0.1 0.1 %s units box
        minimize 1e-6 1e-6 %s %s
        write_data %s_last.data
        """ % (np.random.randint(low=1, high=99999),self.config.sections['MOTION'].nsteps, self.config.sections['MOTION'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

    def box_min(self):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """min_style  cg
        min_modify dmax 0.05 line quadratic
        dump 1 all custom 1 minimize_box.dump id type x y z fx fy fz
        fix box all box/relax iso 0.0 vmax 0.001
        displace_atoms all random 0.1 0.1 0.1 %s units box
        minimize 1e-6 1e-6 %s %s
        write_data %s_last.data""" % (np.random.randint(low=1, high=99999),self.config.sections['MOTION'].nsteps, self.config.sections['MOTION'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

    def run_then_min(self):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """velocity all create %s %s dist gaussian
        fix nve all nve
        fix lan all langevin %s %s 1.0 48279
        run %s
        unfix nve
        unfix lan
        min_style  fire
        min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 100000 halfstepback no initialdelay no
        dump 1 all custom 1 run_minimize.dump id type x y z fx fy fz
        minimize 1e-6 1e-6 %s %s
        write_data %s_last.data""" % (self.config.sections['MOTION'].temperature, np.random.randint(low=1, high=99999), 
                                      self.config.sections['MOTION'].temperature, self.config.sections['MOTION'].temperature, 
                                      self.config.sections['MOTION'].nsteps, self.config.sections['MOTION'].nsteps, 
                                      self.config.sections['MOTION'].nsteps, self.config.sections['MOTION'].nsteps, 
                                      self.config.sections['TARGET'].job_prefix)

        before_score, after_score = Scoring.add_cmds_before_score(add_cmds)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

class Optimize:

    def __init__(self, pt, config, scoring):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.scoring = scoring

    def latin_hyper(self, **kwargs):
        #placeholder for equal sampling accross input space of generated strucutres
        pass

    def sim_anneal(self, **kwargs):
        #placeholder for simulated annealing of generated strucutres
        pass

    def lib_optimizer(self, **kwargs):
        #placeholder, possible for DAKOTA or pyMOO coupling?
        pass
    
    def tournament_selection(self, *args):
        #More of a super function that will call a bunch of the ones below
        #TODO Currently this is a copy/paste of the old code, needs work.
        
        starting_generation = Create.starting_generation()
        scores = []

        for candidate in len(starting_generation):
            scores.append(self.scoring.get_score(starting_generation[candidate]))

        data = np.c_[starting_generation, scores] #appends arrays along the second axis (column-wise)
        
        selection = starting_generation.copy() # copy so we can pop elements out

        for iteration in self.config.sections['GENETIC'].num_generations:               
            # Pick 2 indicies to compare and add the best of to the selection list (e.g. perform a tournament)
            for round in len(selection)-1:
                compare_pair = np.random.randint(0, len(selection), 2)
                if scores[compare_pair[0]] <= scores[compare_pair[1]]:
                    loser = compare_pair[1]
                    selection.pop(loser)
                else:
                    loser = compare_pair[0]
                    selection.pop(loser)
            winner = [iteration, selection, min(scores[compare_pair[0]],scores[compare_pair[1]])]
    
            #Winning candidate is then appended to winners circle list : [generation, ase.Atoms, score]
            try:
                gen_winners = np.c_[gen_winners, winner] #appends arrays along the first axis (row-wise)
            except:
                gen_winners = winner

            #Now setup for the next iteration of the tournament
            scores = [] 

            if np.random.rand() < self.config.sections['GENETIC'].mutation_rate:
                selection = Genetic.mutation(selection) #Will mutation only take in one structure?
            else:
                hybrid_pair = np.random.randint(0, len(gen_winners), 1)
                selection = Genetic.crossover(selection, hybrid_pair) #Should have two structures

            for candidate in len(selection):
                scores.append(self.scoring.get_score(selection[candidate]))


        #End of tournament returns winners circle list to GRS.py -> (convert.ASEtoLAMMPS + write score output)

        return gen_winners
        
    def unique_tournament_selection(self, **kwargs):
        #More of a super function that will call a bunch of the ones below
        #TODO Currently this is a copy/paste of the old code, needs work.

        #This should be the default since we dont want to send duplicates the crossover/mutation
        scores = []
        for candidate in len(population):
            scores.append(Scoring.get_score(population[candidate]))
        selection = []
        for candidate in population:
            if candidate not in selection:
                selection.append(candidate)
        # The structures to chose from should now only contain unique candidates, determined by score        

        # Pick 2 indicies to compare and add the best of
        # to the selection list (e.g. perform a tournament)
        #Allow for a scoring anomoly at some low rate? else return min(scores)?
        for round in len(selection)-1:
            compare_pair = np.random.randint(0, len(selection), 2)
            if scores[compare_pair[0]] <= scores[compare_pair[1]]:
                loser = compare_pair[1]
                selection.pop(loser)
            else:
                loser = compare_pair[0]
                selection.pop(loser)
        if np.random.rand() < self.config.sections['GENETIC'].mutation_rate:
            nextgen_selection = self.mutation(selection) #Will mutation only take in one structure?
        else:
            nextgen_selection = self.crossover(selection) #Should have two structures
        return nextgen_selection

