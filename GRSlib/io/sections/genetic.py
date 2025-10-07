from GRSlib.io.sections.sections import Section

#pt = ParallelTools()
#output = Output()
class Genetic(Section):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['mutation_rate', 'mutation_types', 'population_size', 'ngenerations']
        self._check_section()
        self.mutation_rate = self.get_value("GENETIC", "mutation_rate", 0.5)
        self.mutation_types = self.get_value("GENETIC", "mutation_types", {'perturb_one': 0.5, 'perturb_N' : 0.5, 'flip_one': 0.0, 'flip_N': 0.0, 'create_one' : 0.0, 'delete_one' : 0.0})
        self.population_size = self.get_value("GENETIC", "population_size", 20)
        self.ngenerations = self.get_value("GENETIC", "ngenerations", 10)
        self.delete()
