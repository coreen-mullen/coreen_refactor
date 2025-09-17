from GRSlib.io.sections.sections import Section
#pt = ParallelTools()
#output = Output()
class Scoring(Section):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['moments', 'moments_coeff', 'moment_bonus', 'moments_cross_coeff','moment_cross_bonus',
                            'strength_target','strength_prior','exact_distribution']
        self._check_section()
        self.moments = self.get_value("SCORING", "moments", "mean stdev").split()
        self.moments_coeff = self.get_value("SCORING", "moments_coeff", "1.0 0.1").split()
        self.moment_bonus = self.get_value("SCORING", "moment_bonus", "0.0 0.0").split()
        self.moments_cross_coeff = self.get_value("SCORING", "moments_cross_coeff", "1.0 0.1").split()
        self.moment_cross_bonus = self.get_value("SCORING", "moment_cross_bonus", "0.0 0.0").split()
        self.strength_target = self.get_value("SCORING", "strength_target", 1.0)
        self.strength_prior = self.get_value("SCORING", "strength_prior", 0.0)
        self.exact_distribution = self.get_value("SCORING", "exact_distribution", False)
        self.delete()


