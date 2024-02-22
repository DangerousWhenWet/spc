import math
from typing import Optional, List, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import SPCTrace, draw_spc_matplotlib, draw_spc_plotly
import spc.factors as factors


class PTrace(SPCTrace):
    def __init__(self, data:pd.Series, data_column:str, groupby_column:str, subgroup_size:Optional[int]=None, allow_variable_subgroup_size:bool=False):
        #TODO: allow variable sample sizes
        if not allow_variable_subgroup_size and subgroup_size is None:
            raise ValueError("If you have fixed sample size, you need to provide `sample_size` kwarg")

        self.n = subgroup_size
        self.data = data.groupby(groupby_column)[data_column].mean()
        self.centerline = self.data.mean()
        self.sigma = math.sqrt(  (self.centerline * (1 - self.centerline) ) / self.n )

    def to_plotly(self, xaxis_title:Optional[str]=None, yaxis_title:Optional[str]=None, plotly_theme:Optional[str]='seaborn', fig_kwargs:Optional[Dict]=None, show_weco_rules:Optional[List[int]]=None,):
        fig_kwargs = fig_kwargs or {}
        show_weco_rules = show_weco_rules or []

        y_title = '<b>' + (f"{yaxis_title}, " if yaxis_title else '') + """<span style="text-decoration:overline">x</span>""" + '</b>'
        fig = make_subplots(rows=1,cols=1)

        # Set other cosmetic/presentation stuff
        fig.update_layout(
            template=plotly_theme, margin={'l':0, 'r':0, 't':0, 'b':0}, paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title=y_title
        )
        fig.update_xaxes(type='category', categoryorder='category ascending')
        if xaxis_title:
            fig.update_xaxes(title_text=f"<b>{xaxis_title}</b>")

        # Plot the upper and lower traces
        draw_spc_plotly(fig, self, show_weco_rules, **fig_kwargs)

        return fig


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
