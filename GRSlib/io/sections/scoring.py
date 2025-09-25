from GRSlib.io.sections.sections import Section
#pt = ParallelTools()
#output = Output()
class Scoring(Section):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['score_type','moments', 'moments_coeff', 'moments_bonus', 
                            'strength_target','strength_prior','exact_distribution']
        self._check_section()
        self.score_type = self.get_value("SCORING", "score_type", "moments")
        self.strength_target = self.get_value("SCORING", "strength_target", 1.0)
        self.strength_prior = self.get_value("SCORING", "strength_prior", 0.0)
        self.exact_distribution = self.get_value("SCORING", "exact_distribution", False)
        if self.score_type == "moments":
            self.moments = self.get_value("SCORING", "moments", "mean stdev").split()
            #options are :
            self.moments_coeff = self.get_value("SCORING", "moments_coeff", "1.0 0.1").split()
            #Requires one number per moment
            self.moments_bonus = self.get_value("SCORING", "moments_bonus", "0.0 0.0").split()
            #Score reduction if exact moment value is matched
        self.delete()


