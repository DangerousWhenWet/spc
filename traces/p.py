import math
from typing import Optional, List, Dict, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import SPCTrace, draw_spc_matplotlib, draw_spc_plotly
import spc.factors as factors


class PTrace(SPCTrace):
    def __init__(self, data:pd.Series, data_column:str, groupby_column:str, subgroup_size:Optional[int]=None, allow_variable_subgroup_size:bool=False, reset_grouped_index:bool=False, xaxis_proxy:Optional[pd.Series]=None):
        #TODO: allow variable sample sizes
        if not allow_variable_subgroup_size and subgroup_size is None:
            raise ValueError("If you have fixed sample size, you need to provide `sample_size` kwarg")

        self.n = subgroup_size
        self.data = data.groupby(groupby_column)[data_column].mean()
        if reset_grouped_index:
            self.data.reset_index(drop=True, inplace=True)
        self.centerline = self.data.mean()
        self.sigma = math.sqrt(  (self.centerline * (1 - self.centerline) ) / self.n )
        self.xaxis_proxy = xaxis_proxy

    def to_plotly(
        self,
        xaxis_title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        plotly_theme: Optional[str] = 'seaborn',
        show_weco_rules: Optional[List[int]] = None,
        lsl: Optional[float]=None,
        usl: Optional[float]=None,
        hover_customdata: Optional[List[List[Union[float, pd.Series]]]] = None,
        hover_template: Optional[List[str]] = None,
        **kwargs
    ) -> go.Figure:
        show_weco_rules = show_weco_rules or []

        y_title = '<b>' + (f"{yaxis_title}, " if yaxis_title else '') + """%-conforms""" + '</b>'
        fig = make_subplots(rows=1,cols=1)

        # Set other cosmetic/presentation stuff
        fig.update_layout(
            template=plotly_theme, margin={'l':0, 'r':0, 't':0, 'b':0}, paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title=y_title
        )

        if self.xaxis_proxy is not None:
            proxy_sampled = self.xaxis_proxy.iloc[ np.linspace(0, len(self.xaxis_proxy)-1, num=min(25, len(self.data))) ]
            fig.update_xaxes(
                tickmode='array',
                tickvals=proxy_sampled.index,
                ticktext=proxy_sampled,
            )

        if xaxis_title:
            fig.update_xaxes(title_text=f"<b>{xaxis_title}</b>")

        # Plot the upper and lower traces
        draw_spc_plotly(fig, self, show_weco_rules, clamp_control_limits=(0.0, 1.0), lsl=lsl, usl=usl, hovertemplate=hover_template[0], customdata=hover_customdata[0], **kwargs)

        return fig


    @property
    def traces(self):
        return self,


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
