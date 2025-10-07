from GRSlib.parallel_tools import ParallelTools
from GRSlib.io.sections.sections import Section
from GRSlib.io.sections.error import ExitFunc
from GRSlib.io.sections.basis import Basis
from GRSlib.io.sections.constraint import Constraint
from GRSlib.io.sections.genetic import Genetic
from GRSlib.io.sections.motion import Motion
from GRSlib.io.sections.output import Output
from GRSlib.io.sections.scoring import Scoring
from GRSlib.io.sections.target import Target


#pt = ParallelTools()


def new_section(section, config, pt, infile, args):
    """Section Factory"""
    instance = search(section)
    try:
        instance.__init__(section, config, pt, infile, args)
    except ExitFunc:
        pass
    return instance


def search(section):
    instance = None
    for cls in Section.__subclasses__():
        if cls.__name__.lower() == section.lower():
            instance = Section.__new__(cls)

    if instance is None:
        raise IndexError("{} was not found in grs sections".format(section))
    else:
        return instance
