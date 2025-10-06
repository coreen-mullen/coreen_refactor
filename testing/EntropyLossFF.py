from mpi4py import MPI
from GRSlib.GRS import GRS
import random, copy, os, glob, shutil
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

settings = \
{
"BASIS":
    {
    "descriptor": "ACE",
    "numTypes": 1,
    "elements": "W",
    "rcutfac": 5.5,
    "lambda": 1.4,
    "ranks": "1 2 3",
    "lmax": "0 3 3",
    "lmin": "0 0 0",
    "nmax": "8 1 1",
    "nmaxbase": 8,
    "bzeroflag": 0
    },
"SCORING":
    {
    "score_type": "entropy",
    "internal_entropy": 1.0, 
    "ensemble_entropy": 1.0,
    "strength_target": 0.0,
    "strength_prior": -1.0
    },
"TARGET":
    {
    "target_fname": "bcc.data",
#    "target_fdesc": "fcc.npy",
    "start_fname": "notbcc.data",
    "job_prefix": "GetENT"
    },
"MOTION":
    {
    "soft_strength": 0.5,
    "ml_strength": 1.0,
    "nsteps": 10000,
    "temperature": 0.0,
    "min_type": "line", 
    "randomize_comps": False 
    }
}

grs = GRS(settings,comm=comm)

#grs.config.view_state()

#attributes = [attr for attr in dir(grs.gradient_move) if not attr.startswith('__')]
#print("attr of grs.gradient_move:")
#print(attributes)

#starting_struc = grs.convert_to_desc(settings["TARGET"]["start_fname"])

#score = grs.get_score(settings["TARGET"]["start_fname"])
#print("     Score calculated through LAMMPS:",score)

updated_struct = settings["TARGET"]["start_fname"]
grs.set_prior([updated_struct])

for i in range(10):
    updated_struct = grs.gradient_move(updated_struct)
    updated_struct = grs.update_start(updated_struct,"MaxScore")
    grs.set_prior(glob.glob(settings['TARGET']["job_prefix"]+"*.data"))

exit()

exit()
