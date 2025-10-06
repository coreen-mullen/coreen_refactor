from GRSlib.io.sections.sections import Section

#pt = ParallelTools()
class Output(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['logfiles', 'structures','scores']
        self._check_section()
        self.logfiles = self.get_value("OUTPUT", "logfiles", "out_log")
        self.structures = self.get_value("OUTPUT", "structures", "out_str")
        self.scores = self.get_value("OUTPUT", "scores", "out_score")
        self.delete()

