from GRSlib.parallel_tools import ParallelTools
from ase.io import read, write, lammpsdata


class Convert:

    def __init__(self, name, pt, config):
        self.pt = pt #ParallelTools()
        self.config = config #Config()

#   Ok technically these functions dont do anything right now, but I'm holding onto them as
#   boilerplate for the ASE conversions, or descriptor conversions that dont need LAMMPS

    def ase_to_lammps(self,data):
        """
        Takes in an ase.Atoms object and writes a lammps-data, returns the file name
        """
        fname = '%s.lammps-data' % data.symbols
        write(fname, data, format='lammps-data', masses=True)
        return fname

    def lammps_to_ase(self,data):
        """
        Takes in a lammps-data file and returns an ase.Atoms object
        """
        ase_data = read(data,format='lammps-data')
        return ase_data

    def lammps_ace(self,data):
        """
        Takes in an lammps-data file and converts to descriptors in the ACE basis set
        """
        descriptor_vals = Ace.run_lammps_single(data)
        return descriptor_vals

    def lammps_snap(self,data):
        """
        Takes in an lammps-data file and converts to descriptors in the SNAP basis set (Not Implemented ATM)
        """
        descriptor_vals = Snap.run_lammps_single(data)
        return descriptor_vals

    def lammps_custom(self,data):
        """
        Takes in an lammps-data file and converts to descriptors in a custom basis set (Not Implemented ATM)
        """
        descriptor_vals = Custom.run_lammps_single(data)
        return descriptor_vals

