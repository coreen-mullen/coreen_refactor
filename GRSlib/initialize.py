def initialize_GRS_run():
    try:
        import mpi4py as mpi4py
        from GRS.parallel_tools import ParallelTools
        pt = ParallelTools()

    except ModuleNotFoundError:
        from GRS.parallel_tools import ParallelTools
        pt = ParallelTools()

    except Exception as e:
        print("Trouble importing mpi4py package, exiting...")
        raise e

    try:
        pt.single_print("Reading input...")
        pt.all_barrier()
        from GRS.io.input import Config
        config = Config()
        if (pt._rank==0):
            print(f"Hash: {config.hash}")
        pt.single_print("Finished reading input")
        pt.single_print("------------------")

    except Exception as e:
        pt.single_print("Trouble reading input, exiting...")
        pt.exception(e)

    try:
        pt.single_print("mpi4py version: ", mpi4py.__version__)

    except NameError:
        print("No mpi4py detected, using grs stubs...")

    try:
        import numpy as np
        #output.screen("numpy version: ", np.__version__)
        pt.single_print
    except Exception as e:
        #output.screen("Trouble importing numpy package, exiting...")
        #output.exception(e)
        pt.single_print("Trouble importing numpy package, exiting...")
        pt.single_print(f"{e}")

    try:
        import scipy as sp
        pt.single_print("scipy version: ", sp.__version__)
    except Exception as e:
        pt.single_print("Trouble importing scipy package, exiting...")
        pt.single_print(e)

    try:
        import pandas as pd
        pt.single_print("pandas version: ", pd.__version__)
    except Exception as e:
        pt.single_print("Trouble importing pandas package, exiting...")
        pt.single_print(e)

    try:
        import lammps
        lmp = lammps.lammps()
        #print("LAMMPS version: ",lammps.__version__)
        pt.lammps_version = lammps.__version__
    except Exception as e:
        print("Trouble importing LAMMPS library, exiting...")
        raise e
    
    try:
        import ase as ase
        pt.single_print("ASE version: ", ase.__version__)
    except Exception as e:
        pt.single_print("Trouble importing ASE package, exiting...")
        pt.single_print(e)

    pt.single_print("-----------")
