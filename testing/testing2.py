from time import time
from mpi4py import MPI
from GRSlib.GRS import GRS
import numpy as np
from GRSlib.converters.convert import Convert
from GRSlib.converters.sections.lammps_ace import Ace
#from GRSlib.converters.convert import convert
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
settings = "../GenericInput.in"
grs = GRS(settings, comm=comm)
conv = Convert(name='name',pt=grs.pt, config=grs.config)
ace = Ace(name='name',pt=grs.pt, config=grs.config)
#testing of io class
#grs.config.view_state()
#-----------------------

#testing of convert class
#attributes = [attr for attr in dir(grs.convert) if not attr.startswith('__')]
#print("attr of grs.convert:")
#print(attributes)

atoms = conv.lammps_to_ase('bcc.data')
print(atoms)
file = conv.ase_to_lammps(atoms)
print(file)
descs=ace.run_lammps_single('bcc.data')
print(descs)
#current_desc = convert_to_desc(file)
#-----------------------
#grs.genetic_move.tournament_selection(data=None)

score = grs.get_score('bcc.data')
print(score)
print("!")
exit()