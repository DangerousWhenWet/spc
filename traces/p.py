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
    def __init__(self, data:pd.Series, data_column:str, grouper:str, subgroup_size:Optional[int]=None, allow_variable_subgroup_size:bool=False, reset_grouped_index:bool=False, xaxis_proxy:Optional[pd.Series]=None):
        #TODO: allow variable sample sizes
        if not allow_variable_subgroup_size and subgroup_size is None:
            raise ValueError("If you have fixed sample size, you need to provide `sample_size` kwarg")

        self.k = data.groupby(grouper).size() if allow_variable_subgroup_size else subgroup_size
        self.data = data.groupby(grouper)[data_column].mean()
        if reset_grouped_index:
            self.data.reset_index(drop=True, inplace=True)
        self.centerline = self.data.mean()
        self.sigma = np.sqrt(  (self.centerline * (1 - self.centerline) ) / self.k ) if allow_variable_subgroup_size else math.sqrt( self.centerline * (1 - self.centerline) / self.k )
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
        hover_customdata = hover_customdata or [None]
        hover_template = hover_template or [None]

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
    import random
    random.seed(908)
    np.random.seed(908)

    # generate random subgroup means between given range, random subgroup stddev between given range
    USE_VARIABLE_GROUP_SIZES = True #<-- Edit me to toggle between fixed and variable subgroup sizes
    num_subgroups = 15
    subgroup_means = np.random.uniform(90, 95, size=num_subgroups)
    inaccuracies = np.random.uniform(0, -5, size=num_subgroups)
    subgroup_stddevs = np.random.uniform(1, 20, size=num_subgroups)
    subgroup_sizes = [random.choice(range(50, 100, 1)) for i in range(num_subgroups)] if USE_VARIABLE_GROUP_SIZES else [30]*num_subgroups

    # generate dummy data for each subgroup and then combine it all
    data_per_subgroup = []
    for mean, stddev, size in zip(subgroup_means+inaccuracies, subgroup_stddevs, subgroup_sizes):
        subgroup_data = pd.DataFrame({
            'value': np.random.normal(loc=mean, scale=stddev, size=size),  # Generating 12 values for each subgroup
            'group': np.repeat(len(data_per_subgroup) + 1, repeats=size),  # Assigning group numbers
            'passed': np.nan
        })
        subgroup_data['passed'] = (subgroup_data['value'] > 90) & \
            (subgroup_data['value'] < 95)
        data_per_subgroup.append(subgroup_data)
    data = pd.concat(data_per_subgroup, ignore_index=True)
    print(data)

    p = PTrace(
        data = data,
        data_column = 'passed',
        grouper = 'group',
        allow_variable_subgroup_size = USE_VARIABLE_GROUP_SIZES,
        subgroup_size = subgroup_sizes[0] if not USE_VARIABLE_GROUP_SIZES else None
    )

    fig = p.to_plotly(
        xaxis_title='Group',
        yaxis_title='Pass Rate',
        show_weco_rules=[1],
        line_color='black'
    )
    fig.show()