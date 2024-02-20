import math
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticks

from . import SPCTrace, draw_spc_matplotlib
import spc.factors


class PTrace(SPCTrace):
    def __init__(self, data:pd.Series, sample_size:Optional[int]=None, allow_variable_sample_size:bool=False):
        #TODO: allow variable sample sizes
        if not allow_variable_sample_size and sample_size is None:
            raise ValueError("If you have fixed sample size, you need to provide `sample_size` kwarg")

        if min(data)<0 or max(data)>1.0:
            raise ValueError("You should provide data series in the form 'proportion defective' or 'proportion OK' ranging [0.0, 1.0]")
        self.n = sample_size
        self.data = data
        self.centerline = self.data.mean()
        self.sigma = math.sqrt(  (self.centerline * (1 - self.centerline) ) / self.n )


if __name__ == '__main__':
    #data = pd.Series([50, 60, 10, 50, 40, 30, 20, 50, 50, 60, 150, 125, 70, 80, 60, 70, 60, 80, 60, 50])
    data = pd.Series([0.85, 0.87, 0.80, 0.85, 0.84, 0.83, 0.82, 0.86, 0.80, 0.82, 0.78, 0.79, 1.00, 0.85, 0.84, 0.33, 0.25, 0.80])
    p = PTrace(data, sample_size=30)

    def generate_p(p:PTrace):
        fig, ax = plt.subplots()
        draw_spc_matplotlib(ax, p, 'p')
        ax.xaxis.set_major_locator(mticks.MultipleLocator(5))
        ax.xaxis.set_minor_locator(mticks.MultipleLocator(1))
        ax.set_ylim([-0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]) , ax.get_ylim()[1]])
        return ax
    
    ax = generate_p(p)
    for event in p.get_weco_event_slices(rule_number=1):
        ax.plot(event.index, event, color='red', marker='o', zorder=2)
    plt.show()
