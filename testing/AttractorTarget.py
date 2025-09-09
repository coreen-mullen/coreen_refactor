from mpi4py import MPI
from GRSlib.GRS import GRS
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
    "moments": "mean" ,
    "moments_coeff": "1.0",
    "moment_bonus": "0" ,
    "moments_cross_coeff": "0",
    "moment_cross_bonus": "0",
    "attractor_target": "True",
    "exact_distribution": "False"
    },
"TARGET":
    {
    "target_fname": "bcc.data",
    "start_fname": "notbcc.data"
    },
"MOTION":
    {
    "soft_strength": 1.0,
    "ml_strength": 1.0,
    "nsteps": 1000,
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

score = grs.get_score(settings["TARGET"]["start_fname"])
print("     Mean score calculated through LAMMPS:",score)
#print("Done checking socring!")

grs.gradient_move(settings["TARGET"]["start_fname"])

exit()
