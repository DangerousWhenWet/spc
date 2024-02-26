from typing import Optional, List, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import SPCTrace, draw_spc_matplotlib, draw_spc_plotly
import spc.factors as factors


class XSTraces:
    def __init__(self, data:pd.DataFrame, data_column:str, groupby_column:str, subgroup_size:int, xaxis_proxy:Optional[pd.Series]=None):
        ANTIBIAS_B4 = factors.get_B4(n=subgroup_size)
        ANTIBIAS_A3 = factors.get_A3(n=subgroup_size)
        rational_subgroups = data.groupby(groupby_column)[data_column]
        s = rational_subgroups.std()
        s_bar = s.mean()
        self.s = SPCTrace(
            data = s,
            centerline = s_bar,
            sigma = (ANTIBIAS_B4 * s_bar - s_bar)/3
        )
        x_bar = rational_subgroups.mean()
        grand_mean = x_bar.mean()
        self.x = SPCTrace(
            data = x_bar,
            centerline = grand_mean,
            sigma = (ANTIBIAS_A3 * s_bar)/3
        )
        self.xaxis_proxy = xaxis_proxy

    def to_plotly(self, xaxis_title:Optional[str]=None, yaxis_titles:Optional[List[str]]=None, plotly_theme:Optional[str]='seaborn', fig_kwargs:Optional[Dict]=None, show_weco_rules:Optional[List[int]]=None, force_categorical:bool=False):
        fig_kwargs = fig_kwargs or {}
        show_weco_rules = show_weco_rules or []

        y1_title = '<b>' + (f"{yaxis_titles[0]}, " if yaxis_titles else '') + """<span style="text-decoration:overline">x</span>""" + '</b>'
        y2_title = '<b>' + (f"{yaxis_titles[1]}, " if yaxis_titles else '') + r"S" + '</b>'
        fig = make_subplots( rows=2, row_heights=[0.5, 0.5], cols=1, shared_xaxes=True, vertical_spacing=0.02,)

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
                row=2, col=1
            )
        if force_categorical:
            fig.update_xaxes(type='category', categoryorder='category ascending')
        if xaxis_title:
            fig.update_xaxes(title_text=f"<b>{xaxis_title}</b>", row=2)

        # Plot the upper and lower traces
        for trace, row_number in ((self.x, 1), (self.s, 2)):
            draw_spc_plotly(fig, trace, show_weco_rules, row_number, **fig_kwargs)

        return fig

if __name__ == '__main__':
    np.random.seed(908)

    # generate random subgroup means between 70-90, random subgroup stddev between 5-30
    num_subgroups = 20
    subgroup_means = np.random.uniform(85, 95, size=num_subgroups)
    subgroup_stddevs = np.random.uniform(1, 5, size=num_subgroups)
    subgroup_size = 30

    # generate dummy data for each subgroup and then combine it all
    data_per_subgroup = []
    for mean, stddev in zip(subgroup_means, subgroup_stddevs):
        subgroup_data = pd.DataFrame({
            'value': np.random.normal(loc=mean, scale=stddev, size=subgroup_size),  # Generating 12 values for each subgroup
            'group': np.repeat(len(data_per_subgroup) + 1, repeats=subgroup_size)  # Assigning group numbers
        })
        data_per_subgroup.append(subgroup_data)
    data = pd.concat(data_per_subgroup, ignore_index=True)
    with pd.option_context('display.max_rows', None):
        print(data)

    xs = XSTraces(data, data_column='value', groupby_column='group', subgroup_size=subgroup_size)

    def generate_xs(xs:XSTraces):
        fig, (ax_x, ax_s) = plt.subplots(nrows=2, ncols=1, sharex=True)
        _ = plt.subplots_adjust(hspace=0)
        draw_spc_matplotlib(ax_x, xs.x, 'x̅')
        draw_spc_matplotlib(ax_s, xs.s, 'S')
        ax_s.xaxis.set_major_locator(mticks.MultipleLocator(5))
        ax_s.xaxis.set_minor_locator(mticks.MultipleLocator(1))
        ax_s.set_ylim([-0.05 * (ax_s.get_ylim()[1] - ax_s.get_ylim()[0]) , ax_s.get_ylim()[1]])

        return ax_x, ax_s
    
    ax_x, ax_s = generate_xs(xs)
    for event in xs.x.get_weco_event_slices(rule_number=2):
        ax_x.plot(event.index, event, color='red', marker='o', zorder=2)
    for event in xs.s.get_weco_event_slices(rule_number=2):
        ax_s.plot(event.index, event, color='red', marker='o', zorder=2)
    plt.show()
