"""
Library implementing different objective functions.
"""

from operator import mul

MINIMIZE = "minimize"
MAXIMIZE = "maximize"

class TargetVarDictObjective(object):
    """
    This is the general objective, which takes an output of the network and pushes it to a certain label
    """
    # Is the optimal high or low? Which way to optimize?
    optimize = MINIMIZE
    # Can you take the loss per sample, and average later? Or is this like AUC, where you need all outputs and labels?
    mean_over_samples = True

    def __init__(self, *args, **kwargs):
        # create target vars, but only if super class doesn't have them
        try:
            self.target_vars
        except:
            self.target_vars = dict()

    def get_loss(self, *args, **kwargs):
        """Compute the loss in Theano."""
        raise NotImplementedError

    def get_loss_from_lists(self, predicted, expected, *args, **kwargs):
        """Compute the loss in Numpy (for validation purposes usually)."""
        raise NotImplementedError

    def __add__(self, other):
        return SumObjectives(self, other)

    def __radd__(self, other):
        return SumObjectives(other, self)

    def __mul__(self, other):
        return MultiplyObjectives(self, other)

    def __rmul__(self, other):
        return MultiplyObjectives(other, self)


class SumObjectives(TargetVarDictObjective):
    def __init__(self, *objectives):
        super(SumObjectives,self).__init__()
        self.input_objectives = objectives
        for obj in self.input_objectives:
            if isinstance(obj, TargetVarDictObjective):
                self.target_vars.update(obj.target_vars)
                obj.target_vars = self.target_vars

    def get_loss(self, *args, **kwargs):
        r = [obj.get_loss(*args, **kwargs) if isinstance(obj, TargetVarDictObjective) else obj for obj in self.input_objectives]
        return sum(r)


class MultiplyObjectives(TargetVarDictObjective):
    def __init__(self, *objectives):
        super(MultiplyObjectives,self).__init__()
        self.input_objectives = objectives
        for obj in self.input_objectives:
            if isinstance(obj, TargetVarDictObjective):
                self.target_vars.update(obj.target_vars)
                obj.target_vars = self.target_vars

    def get_loss(self, *args, **kwargs):
        r = [obj.get_loss(*args, **kwargs) if isinstance(obj, TargetVarDictObjective) else obj for obj in self.input_objectives]
        return reduce(mul, r)
