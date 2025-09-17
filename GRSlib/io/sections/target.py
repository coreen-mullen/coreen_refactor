from GRSlib.io.sections.sections import Section

#pt = ParallelTools()
#output = Output()
class Target(Section):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['target_fname', 'target_fdesc', 'start_fname', 'job_prefix']
        self._check_section()
        self.target_fname = self.get_value("TARGET", "target_fname", None)
        self.target_fdesc = self.get_value("TARGET", "target_fdesc", None)
        self.start_fname = self.get_value("TARGET", "start_fname", None)
        self.job_prefix = self.get_value("TARGET", "job_prefix", "GRSOutput")
        self.delete()
