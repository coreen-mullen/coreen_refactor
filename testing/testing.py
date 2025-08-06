from time import time
from mpi4py import MPI
from GRSlib.GRS import GRS
import numpy as np

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

#atoms = grs.convert.lammps_to_ase('bcc.data')
#print(atoms)
#file = grs.convert.ase_to_lammps(atoms)
#print(file)
#grs.convert.run_lammps_single('bcc.data')
#current_desc = grs.convert_to_desc(file)
#-----------------------
#grs.genetic_move.tournament_selection(data=None)

score = grs.get_score('bcc.data')
print(score)
#print("!")
exit()
