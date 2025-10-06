from GRSlib.io.sections.sections import Section

#pt = ParallelTools()
#output = Output()
class Constraint(Section):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['masses','target_comps','minatoms','maxatoms','template']
        self._check_section()
        self.masses = self.get_value("CONSTRAINT", "masses", {1: 1.004})
        self.target_comps = self.get_value("CONSTRAINT", "target_comps", {'H':1.0} )
        self.minatoms = self.get_value("CONSTRAINT", "minatoms", 27)
        self.maxatoms = self.get_value("CONSTRAINT", "maxatoms", 28)
        self.template = self.get_value("CONSTRAINT", "template", None)
        self.delete()
