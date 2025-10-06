from time import time
from mpi4py import MPI
from GRSlib.GRS import GRS
import numpy as np
from GRSlib.converters.convert import Convert
from GRSlib.converters.sections.lammps_ace import Ace
#from GRSlib.converters.convert import convert
from ase.build import bulk
from ase.io import read,write

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
atoms = bulk('W','bcc',cubic=True)
write('simple_bcc.data',atoms,format='lammps-data')

#grs.current_desc = current_desc
#grs.random_values = np.random.rand(*current_desc.shape)
print('testing')
#grs.genetic_move('bcc.data')

score = grs.get_score('bcc.data')

#ensemb = grs.get_ensemble('bcc.data')
#print(ensemb)

print("!!!")
ensemble_scores = grs.get_ensemble('bcc.data')
print("Ensemble Scores:", ensemble_scores)



#print(score)
#print("!")

#print('current desc',current_desc)
#score = grs.get_score('bcc.data')
#print(score)
#print("!")
exit()
