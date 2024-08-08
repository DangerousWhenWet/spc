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
    def __init__(self, data:pd.DataFrame, data_column:str, grouper:str, subgroup_size:Optional[int]=None, reset_grouped_index:bool=False, xaxis_proxy:Optional[pd.Series]=None):
        #TODO: allow variable sample sizes
        allow_variable_subgroup_size = subgroup_size is None

        self.n = data.groupby(grouper).size() if allow_variable_subgroup_size else subgroup_size
        self.data = data.groupby(grouper)[data_column].mean()
        if reset_grouped_index:
            self.data.reset_index(drop=True, inplace=True)
        self.centerline = self.data.mean()
        self.sigma = np.sqrt(  (self.centerline * (1 - self.centerline) ) / self.n ) if allow_variable_subgroup_size else math.sqrt( self.centerline * (1 - self.centerline) / self.n )
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
        fig = make_subplots(rows=1,cols=1, specs=[[{'secondary_y': True}]])

        # Set other cosmetic/presentation stuff
        fig.update_layout(
            template=plotly_theme, margin={'l':0, 'r':0, 't':0, 'b':0}, paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title=y_title
        )
        fig.update_yaxes(range=[-0.05, 1.05], tickformat='.1%')

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
        # HACK: for plotly.js-related reasons (???), appending the element `<extra></extra>` to the hovertemplate string will prevent hoverlabel
        # from displaying the index of the trace when hovermode is set to 'x unified' or 'y unified'.
        HOVERHACK = '<extra></extra>'
        draw_spc_plotly(fig, self, show_weco_rules, clamp_control_limits=(0.0, 1.0), lsl=lsl, usl=usl, hovertemplate=hover_template[0] + HOVERHACK, customdata=hover_customdata[0], **kwargs)

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
    num_subgroups = 30
    subgroup_means = np.random.uniform(90, 95, size=num_subgroups)
    biases = np.random.uniform(0, -5, size=num_subgroups)
    subgroup_stddevs = np.random.uniform(1, 20, size=num_subgroups)
    subgroup_sizes = [random.choice(range(50, 100, 1)) for i in range(num_subgroups)] if USE_VARIABLE_GROUP_SIZES else [30]*num_subgroups

    # generate dummy data for each subgroup and then combine it all
    data_per_subgroup = []
    for mean, stddev, size in zip(subgroup_means+biases, subgroup_stddevs, subgroup_sizes):
        subgroup_data = pd.DataFrame({
            'value': np.random.normal(loc=mean, scale=stddev, size=size),
            'group': np.repeat(len(data_per_subgroup) + 1, repeats=size),
            'passed': np.nan
        })
        subgroup_data['passed'] = (subgroup_data['value'] > 85) & \
            (subgroup_data['value'] < 95)
        data_per_subgroup.append(subgroup_data)
    data = pd.concat(data_per_subgroup, ignore_index=True)
    print(data)

    p = PTrace(
        data = data,
        data_column = 'passed',
        grouper = 'group',
        subgroup_size = subgroup_sizes[0] if not USE_VARIABLE_GROUP_SIZES else None
    )

    fig = p.to_plotly(
        xaxis_title='Group',
        yaxis_title='Pass Rate',
        show_weco_rules=[4],
        line_color='black', mode='markers+lines', line_width=2
    )
    fig.show()