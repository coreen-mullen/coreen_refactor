from ase import Atoms,Atom
from ase.io import read,write
from ase.ga.utilities import closest_distances_generator, CellBounds
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.neighborlist import primitive_neighbor_list

# Lowest level functions that can be used to modif structures, inherited class not needed since scoring will happen in
# motion/genetic.py. This collection of functions is mostly to avoid clustter and massive files where more abstract 
# things are happening.

def flip_one_atom(atoms,types,endpoint_compositions=False):
    #TODO Currently this is a copy/paste of the old code, needs work.
    if endpoint_compositions:
        new_atoms = atoms.copy()
        flip_ind = np.random.randint(0,len(atoms))
        flip_current = new_atoms[flip_ind].symbol
        excluded = [typ for typ in types if typ != flip_current]
        flip_to_ind = np.random.randint(0,len(excluded))
        flip_to_type = excluded[flip_to_ind]
        new_atoms[flip_ind].symbol = flip_to_type
    else:
        itr = 0
        comps_dct = get_comp(atoms,types)
        comp_vals = list(comps_dct.values())
        while itr == 0 or any([icomp == 0.0 for icomp in comp_vals]):
            new_atoms = atoms.copy()
            flip_ind = np.random.randint(0,len(atoms))
            flip_current = new_atoms[flip_ind].symbol
            excluded = [typ for typ in types if typ != flip_current]
            flip_to_ind = np.random.randint(0,len(excluded))
            flip_to_type = excluded[flip_to_ind]
            new_atoms[flip_ind].symbol = flip_to_type
            comps_dct = get_comp(new_atoms,types)
            comp_vals = list(comps_dct.values())
            itr += 1
    return new_atoms

def flip_N_atoms(atoms,types,fraction=None,endpoint_compositions=False):
    #TODO Currently this is a copy/paste of the old code, needs work.
    if endpoint_compositions:
        fraction = np.random.rand()
        pert_inds = np.random.choice(range(len(atoms)),size=int(len(atoms)*fraction) )
        new_atoms = atoms.copy()
        for pert_ind in pert_inds:
            flip_ind = np.random.randint(0,len(atoms))
            flip_current = new_atoms[flip_ind].symbol
            excluded = [typ for typ in types if typ != flip_current]
            flip_to_ind = np.random.randint(0,len(excluded))
            flip_to_type = excluded[flip_to_ind]
            new_atoms[flip_ind].symbol = flip_to_type
    else:
        comps_dct = get_comp(atoms,types)
        comp_vals = list(comps_dct.values())
        itr = 0
        while itr == 0 or any([icomp == 0.0 for icomp in comp_vals]):
            fraction = np.random.rand()
            pert_inds = np.random.choice(range(len(atoms)),size=int(len(atoms)*fraction) )
            new_atoms = atoms.copy()
            for pert_ind in pert_inds:
                flip_ind = np.random.randint(0,len(atoms))
                flip_current = new_atoms[flip_ind].symbol
                excluded = [typ for typ in types if typ != flip_current]
                flip_to_ind = np.random.randint(0,len(excluded))
                flip_to_type = excluded[flip_to_ind]
                new_atoms[flip_ind].symbol = flip_to_type
            comps_dct = get_comp(new_atoms,types)
            comp_vals = list(comps_dct.values())
            itr += 1

    return new_atoms
