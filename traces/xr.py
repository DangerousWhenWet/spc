from typing import Optional, List, Dict, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import SPCTrace, draw_spc_matplotlib, draw_spc_plotly
import spc.factors as factors


class XRTraces: 
    def __init__(self, data:pd.DataFrame, data_column:str, grouper:str, subgroup_size:int, xaxis_proxy:Optional[pd.Series]=None, reset_grouped_index=False):
        allow_variable_subgroup_size = subgroup_size is None

        self.n = data.groupby(grouper).size() if allow_variable_subgroup_size else subgroup_size
        rational_subgroups = data.groupby(grouper)[data_column]
        r = rational_subgroups.max() - rational_subgroups.min()
        r_bar = r.mean()
        antibias_D4 = self.n.apply(factors.get_D4) if allow_variable_subgroup_size else factors.get_D4(n=self.n)
        self.r = SPCTrace(
            data = r,
            centerline = r_bar,
            sigma = (antibias_D4 * r_bar - r_bar)/3
        )

        x_bar = rational_subgroups.mean()
        grand_mean = x_bar.mean()
        antibias_A2 = self.n.apply(factors.get_A2) if allow_variable_subgroup_size else factors.get_A2(n=self.n)
        self.x = SPCTrace(
            data = x_bar,
            centerline = grand_mean,
            sigma = (antibias_A2 * r_bar)/3
        )

        if reset_grouped_index:
            self.x.data.reset_index(drop=True, inplace=True)
            self.r.data.reset_index(drop=True, inplace=True)
            self.n.reset_index(drop=True, inplace=True)
        self.r.n = self.n
        self.x.n = self.n
        self.xaxis_proxy = xaxis_proxy


    @property
    def traces(self):
        return self.x, self.r


    def to_plotly(
        self,
        xaxis_title: Optional[str] = None,
        yaxis_titles: Optional[List[str]]=None,
        plotly_theme: Optional[str] = 'seaborn',
        show_weco_rules: Optional[List[int]] = None,
        lsl: Optional[float]=None,
        usl: Optional[float]=None,
        hover_customdata: Optional[List[List[Union[float, pd.Series]]]] = None,
        hover_template: Optional[List[str]] = None,
        **kwargs
    ) -> go.Figure:
        show_weco_rules = show_weco_rules or []
        hover_customdata = hover_customdata or [None, None]
        hover_template = hover_template or [None, None]

        y1_title = '<b>' + (f"{yaxis_titles[0]}, " if yaxis_titles else '') + """<span style="text-decoration:overline">x</span>""" + '</b>'
        y2_title = '<b>' + (f"{yaxis_titles[1]}, " if yaxis_titles else '') + r"R" + '</b>'
        fig = make_subplots(
            rows=2, row_heights=[0.5, 0.5], cols=1, shared_xaxes=True, vertical_spacing=0.02,
        )

        # Set other cosmetic/presentation stuff
        fig.update_layout(
            template=plotly_theme, margin={'l':0, 'r':0, 't':0, 'b':0}, paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title=y1_title, yaxis2_title=y2_title,
        )

        if self.xaxis_proxy is not None:
            proxy_sampled = self.xaxis_proxy.iloc[ np.linspace(0, len(self.xaxis_proxy)-1, num=min(25, len(self.x.data))) ]
            fig.update_xaxes(
                tickmode='array',
                tickvals=proxy_sampled.index,
                ticktext=proxy_sampled,
            )

        if xaxis_title:
            fig.update_xaxes(title_text=f"<b>{xaxis_title}</b>", row=2)

        # Plot the upper and lower traces
        for trace, row_number, clamp_limits, lsl, usl in [(self.x, 1, None, lsl, usl), (self.r, 2, (0, float('Infinity')), None, None)]:
            draw_spc_plotly(fig, trace, show_weco_rules, row_number, clamp_control_limits=clamp_limits, lsl=lsl, usl=usl, hovertemplate=hover_template[row_number-1], customdata=hover_customdata[row_number-1], **kwargs)

        return fig


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
        })
        data_per_subgroup.append(subgroup_data)
    data = pd.concat(data_per_subgroup, ignore_index=True)
    print(data)

    xr = XRTraces(
        data = data,
        data_column = 'value',
        grouper = 'group',
        subgroup_size = subgroup_sizes[0] if not USE_VARIABLE_GROUP_SIZES else None
    )
    fig = xr.to_plotly(
        xaxis_title='Group',
        yaxis_titles=['Value', 'Range'],
        show_weco_rules=[2,3],
        line_color='black', mode='markers+lines', line_width=2
    )
    fig.show()
