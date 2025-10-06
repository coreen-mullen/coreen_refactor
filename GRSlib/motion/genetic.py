import numpy as np
#from ase.build import bulk
#from ase.io import read,write
#from ase.ga.utilities import closest_distances_generator, CellBounds
#from ase.ga.startgenerator import StartGenerator
#from ase import Atoms,Atom
#from ase.data import atomic_numbers

class Genetic:
#TODO need to make heavy use of converting between ASE and LAMMPS-Data objects
#TODO each function call here needs to end with scoring call

    def __init__(self, pt, config, scoring):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.scoring = scoring


    def crossover(parent1, parent2, **kwargs):
        #TODO Currently this is a copy/paste of the old code, needs work.
        if target_comps or types==None:
            assert len(parent1) == len(parent2), "parents must have the same length"
            #TODO Need a way to propose another parent, or force a choice of size
            #Call back to tourny selection and get a new candidate
            psize = len(parent1)
            if inputseed != None:
                np.random.seed(inputseed)
            cross_point = np.random.randint(1, psize-1)
            child1 = parent1[:cross_point] + parent2[cross_point:]
            child2 = parent2[:cross_point] + parent1[cross_point:]
        else:
            assert len(parent1) == len(parent2), "parents must have the same length"
            #TODO Need a way to propose another parent, or force a choice of size
            #Call back to tourny selection and get a new candidate
            comps_dct1 = self.get_comp(parent1,types)
            comp_vals1 = list(comps_dct1.values())
            comps_dct2 = self.get_comp(parent2,types)
            comp_vals2 = list(comps_dct2.values())
            while itr == 0 or any([icomp == 0.0 for icomp in comp_vals1]) or any([icomp == 0.0 for icomp in comp_vals2]):
                psize = len(parent1)
                if inputseed != None:
                    np.random.seed(inputseed)
                cross_point = np.random.randint(1, psize-1)
                child1 = parent1[:cross_point] + parent2[cross_point:]
                child2 = parent2[:cross_point] + parent1[cross_point:]
                comps_dct1 = self.get_comp(child1,types)
                comp_vals1 = list(comps_dct1.values())
                comps_dct2 = self.get_comp(child2,types)
                comp_vals2 = list(comps_dct2.values())
                itr += 1
        return [child1, child2]

    def mutation(self, **kwargs):
        #TODO Currently this is a copy/paste of the old code, needs work.
        # TODO from config, find the set of choices, roll dice and then call down to density/perturbation/alchemy
        chosen_mutation  = 'add_atom'
        (chosen_mutation)()


        if 'flip' in mutation_type:
            return mutation_types[mutation_type](current_atoms,types)
        else:
            return mutation_types[mutation_type](current_atoms,scale)
        

        #This def can be cleaner, with a func call dependent on choice
        #if '' == add_atom:
        #    density.add_atom()
        #if '' == del_atom:
        #    density.remove_atom()
        #if '' == change_cell:
        #    density.change_cell()
        # ...

    def get_comp(self, atoms, symbols):
        comps = {symbol: 0.0 for symbol in symbols}
        counts = {symbol: 0 for symbol in symbols}
        atsymbols = [atom.symbol for atom in atoms]
        for atsymbol in atsymbols:
            counts[atsymbol] +=1
        for symbol in symbols:
            comps[symbol] = counts[symbol]/len(atoms)
        return comps


    def mutation_type_from_prob(choice_probs):
        choices = list(choice_probs.keys())
        probs = list(choice_probs.values())
        mut_typ = np.random.choice(choices,p=probs)
        return mut_typ

    def generate_random_integers(sum_value, n):
        # Generate a list of n - 1 random integers
        random_integers = [np.random.randint(0, sum_value) for _ in range(n - 1)]
        # Ensure the sum of the random integers is less than or equal to sum_value
        random_integers.sort()
        # Calculate the Nth integer to ensure the sum is equal to sum_value
        random_integers.append(sum_value - sum(random_integers))
        
        return random_integers

    def generate_occs(target, natoms):
        if type(target) == dict:
            counts = {k: round(v * natoms) for k, v in target.items()}
            diff = natoms - sum(counts.values())
            while diff != 0:
                for k in counts:
                    if diff > 0:
                        counts[k] += 1
                        diff -= 1
                    elif diff < 0 and counts[k] > 0:
                        counts[k] -= 1
                        diff += 1
            occs = [k for k, v in counts.items() for _ in range(v)]
            np.random.shuffle(occs)
        elif type(target) == list:
            occs = np.random.choice(target,natoms).tolist()
        return occs

