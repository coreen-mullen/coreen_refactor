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
    "score_type": "moments",
    "strength_target": 1.0, 
    "strength_prior": 0.0, 
    "moments": "mean stdev" ,
    "moments_coeff": "1.0 0.1",
    "moments_bonus": "0 0" ,
    },
"TARGET":
    {
    "target_fname": "bcc.data",
    "target_fdesc": "bcc.npy",
    "start_fname": "notbcc.data",
    "job_prefix": "TrialGRS"
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

#testing of io class
#grs.config.view_state()
#-----------------------

#testing of convert class
#attributes = [attr for attr in dir(grs.convert) if not attr.startswith('__')]
#print("attr of grs.convert:")
#print(attributes)
#current_desc = grs.convert_to_desc('bcc.data')
#grs.genetic_move.tournament_selection(data=None)

#score = grs.get_score(settings["TARGET"]["start_fname"])
#print("     Score calculated through LAMMPS:",score)
#print("Done checking socring!")

updated_struct = settings["TARGET"]["start_fname"]
grs.set_prior([updated_struct])

for i in range(10):
#    grs.update_prior()
    updated_struct = grs.gradient_move(updated_struct)
    updated_struct = grs.update_start(updated_struct,"MinScore")
    grs.set_prior(glob.glob(settings['TARGET']["job_prefix"]+"*.data"))

exit()
