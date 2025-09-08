from time import time
from mpi4py import MPI
from GRSlib.GRS import GRS
import numpy as np
from ase.build import bulk
from ase.io import read,write

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

settings = "../GenericInput.in"
grs = GRS(settings, comm=comm)

#testing of io class
#grs.config.view_state()
#-----------------------

#testing of convert class
#attributes = [attr for attr in dir(grs.convert) if not attr.startswith('__')]
#print("attr of grs.convert:")
#print(attributes)

atoms = bulk('W','bcc',cubic=True)
write('simple_bcc.data',atoms,format='lammps-data')

#atoms = grs.convert.lammps_to_ase('bcc.data')
#print(atoms)
#file = grs.convert.ase_to_lammps(atoms)
#print(file)
#grs.convert.run_lammps_single('bcc.data')
#current_desc = grs.convert_to_desc(file)
#-----------------------
#grs.genetic_move.tournament_selection(data=None)

grs.current_desc = current_desc
grs.random_values = np.random.rand(*current_desc.shape)
print('testing')
grs.genetic_move('bcc.data')

score = grs.get_score('bcc.data')

ensemb = grs.ensemble_score('bcc.data')
print(ensemb)
print(score)
#print("!")

print('current desc',current_desc)
#score = grs.get_score('bcc.data')
#print(score)
print("!")
exit()
