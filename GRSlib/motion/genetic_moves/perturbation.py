import ase
from ase import Atoms,Atom
from ase.io import read,write
from ase.ga.utilities import closest_distances_generator, CellBounds
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.neighborlist import primitive_neighbor_list

# Lowest level functions that can be used to modif structures, inherited class not needed since scoring will happen in
# motion/genetic.py. This collection of functions is mostly to avoid clustter and massive files where more abstract 
# things are happening.

def perturb_one_atom(atoms,scale=0.5,max_attempt=100,apply_to='ase'):
    #TODO Currently this is a copy/paste of the old code, needs work.
    if apply_to == 'ase':
        cutoffs = ase.neighborlist.natural_cutoffs(atoms)
        sym_num_map = {sym:atomic_numbers[sym] for sym in atoms.symbols}
        #nl = ase.neighborlist.neighbor_list('ijd', atoms, max(cutoffs))
        nl = ase.neighborlist.neighbor_list('ijd', atoms, cutoffs)
        #nl.update(atoms)
        #indices, offsets = nl.get_neighbors(0)`
        Zs = [ atomic_numbers[sym] for sym in list(set(atoms.symbols))]
        blmin = closest_distances_generator(atom_numbers=Zs,
                                            ratio_of_covalent_radii=0.5)
        good_pert = False
        nattempt =0
        new_atoms = atoms.copy()
        while not good_pert and nattempt < max_attempt:
            new_atoms = atoms.copy()
            pert_ind = np.random.randint(0,len(atoms))
            perturbation = np.random.rand(1,3)[0]
            posneg = 2.*(perturbation - np.min(perturbation))/np.ptp(perturbation)-1
            posneg *= scale
            new_atoms[pert_ind].x += posneg[0]
            new_atoms[pert_ind].y += posneg[1]
            new_atoms[pert_ind].z += posneg[2]
            nl = ase.neighborlist.neighbor_list('ijd', new_atoms, cutoffs)
            sym_num_map = {sym:atomic_numbers[sym] for sym in new_atoms.symbols}
            pair_dist_flags = [ blmin[tuple(sorted([sym_num_map[new_atoms.symbols[i]], sym_num_map[ new_atoms.symbols[nl[1][i_ind]]] ]))] > nl[2][i_ind] for i_ind, i in enumerate(nl[0])]
            if not any(pair_dist_flags):
                good_pert = True
            nattempt += 1
        return new_atoms

        """
        new_atoms = atoms.copy()
        pert_ind = np.random.randint(0,len(atoms))
        perturbation = np.random.rand(1,3)[0]
        posneg = 2.*(perturbation - np.min(perturbation))/np.ptp(perturbation)-1
        posneg *= scale
        new_atoms[pert_ind].x += posneg[0]
        new_atoms[pert_ind].y += posneg[1]
        new_atoms[pert_ind].z += posneg[2]
        return new_atoms
        """

    elif apply_to == 'raw_positions':
        new_atoms = atoms.copy()
        pert_ind = np.random.randint(0,len(atoms))
        perturbation = np.random.rand(1,3)[0]
        posneg = 2.*(perturbation - np.min(perturbation))/np.ptp(perturbation)-1
        posneg *= scale
        new_atoms[pert_ind][0] += posneg[0]
        new_atoms[pert_ind][1] += posneg[1]
        new_atoms[pert_ind][2] += posneg[2]
        return new_atoms

def perturb_N_atoms(atoms,scale=0.5,max_attempt = 100, fraction=0.25):
    #TODO Currently this is a copy/paste of the old code, needs work.
    cutoffs = ase.neighborlist.natural_cutoffs(atoms)
    sym_num_map = {sym:atomic_numbers[sym] for sym in atoms.symbols}
    #nl = ase.neighborlist.neighbor_list('ijd', atoms, max(cutoffs))
    nl = ase.neighborlist.neighbor_list('ijd', atoms, cutoffs)
    #nl.update(atoms)
    #indices, offsets = nl.get_neighbors(0)`
    Zs = [ atomic_numbers[sym] for sym in list(set(atoms.symbols))]
    blmin = closest_distances_generator(atom_numbers=Zs,
                                        ratio_of_covalent_radii=0.5)

    good_pert = False
    nattempt =0
    new_atoms = atoms.copy()
    while not good_pert and nattempt < max_attempt:
        pert_inds = np.random.choice(range(len(atoms)),size=int(len(atoms)*fraction) )
        for pert_ind in pert_inds:
            new_atoms = atoms.copy()
            perturbation = np.random.rand(1,3)[0]
            posneg = 2.*(perturbation - np.min(perturbation))/np.ptp(perturbation)-1
            posneg *= scale
            new_atoms[pert_ind].x += posneg[0]
            new_atoms[pert_ind].y += posneg[1]
            new_atoms[pert_ind].z += posneg[2]
        nl = ase.neighborlist.neighbor_list('ijd', new_atoms, cutoffs)
        sym_num_map = {sym:atomic_numbers[sym] for sym in new_atoms.symbols}
        pair_dist_flags = [ blmin[tuple(sorted([sym_num_map[new_atoms.symbols[i]], sym_num_map[ new_atoms.symbols[nl[1][i_ind]]] ]))] > nl[2][i_ind] for i_ind, i in enumerate(nl[0])]
        if not any(pair_dist_flags):
            good_pert = True
        nattempt += 1
    if not good_pert:
        print ("WARNING: this mutation has a bad neighbor distance - try changing your scale parameter")
    return new_atoms
