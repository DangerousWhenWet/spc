from typing import Optional, List, Dict, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import SPCTrace, draw_spc_matplotlib, draw_spc_plotly
import spc.factors as factors


class XMRTraces:
    def __init__(self, data:pd.Series, xaxis_proxy:Optional[pd.Series]=None):
        ANTIBIAS_D4 = factors.get_D4(n=2)
        ANTIBIAS_d2 = factors.get_d2(n=2)

        mr = data.diff().abs()
        mr_bar = mr.dropna().mean()
        self.mr = SPCTrace(
            data = mr,
            centerline = mr_bar,
            sigma = (ANTIBIAS_D4 * mr_bar - mr_bar)/3,
            name = 'mR'
        )
        self.x = SPCTrace(
            data = data,
            centerline = data.mean(),
            sigma = (3/ANTIBIAS_d2 * mr_bar)/3,
            name = 'X'
        )
        self.xaxis_proxy = xaxis_proxy
    
    @property
    def traces(self):
        return self.x, self.mr


    def to_plotly(
            self,
            xaxis_title:Optional[str]=None,
            yaxis_titles:Optional[List[str]]=None,
            plotly_theme:Optional[str]='seaborn',
            show_weco_rules:Optional[List[int]]=None, 
            force_categorical:bool=False,
            lsl: Optional[float]=None,
            usl: Optional[float]=None,
            hover_customdata: Optional[List[List[Union[float, pd.Series]]]] = None,
            hover_template: Optional[List[str]] = None,
            **kwargs
        ):
        show_weco_rules = show_weco_rules or []

        y1_title = '<b>' + (f"{yaxis_titles[0]}, " if yaxis_titles else '') + """X""" + '</b>'
        y2_title = '<b>' + (f"{yaxis_titles[1]}, " if yaxis_titles else '') + r"mR" + '</b>'
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
        # HACK: for plotly.js-related reasons (???), appending the element `<extra></extra>` to the hovertemplate string will prevent hoverlabel
        # from displaying the index of the trace when hovermode is set to 'x unified' or 'y unified'.
        HOVERHACK = '<extra></extra>'
        for trace, row_number, clamp_limits, lsl, usl in [(self.x, 1, None, lsl, usl), (self.mr, 2, (0, float('Infinity')), None, None)]:
            draw_spc_plotly(fig, trace, show_weco_rules, row_number, lsl=lsl, usl=usl, clamp_control_limits=clamp_limits, customdata=hover_customdata[row_number-1], hovertemplate=hover_template[row_number-1] + HOVERHACK, **kwargs)

        return fig


if __name__ == '__main__':
    data = pd.Series([50, 60, 10, 50, 40, 30, 20, 50, 50, 60, 150, 125, 70, 80, 60, 70, 60, 80, 60, 50])
    xmr = XMRTraces(data)

    def generate_xmr(xmr:XMRTraces):
        fig, (ax_x, ax_mr) = plt.subplots(nrows=2, ncols=1, sharex=True)
        _ = plt.subplots_adjust(hspace=0)
        draw_spc_matplotlib(ax_x, xmr.x, 'X')
        draw_spc_matplotlib(ax_mr, xmr.mr, 'mR')
        ax_mr.xaxis.set_major_locator(mticks.MultipleLocator(5))
        ax_mr.xaxis.set_minor_locator(mticks.MultipleLocator(1))
        ax_mr.set_ylim([-0.05 * (ax_mr.get_ylim()[1] - ax_mr.get_ylim()[0]) , ax_mr.get_ylim()[1]])

        return ax_x, ax_mr
    
    ax_x, ax_mr = generate_xmr(xmr)
    for event in xmr.x.get_weco_event_slices(rule_number=1):
        ax_x.plot(event.index, event, color='red', marker='o', zorder=2)
    for event in xmr.mr.get_weco_event_slices(rule_number=1):
        ax_mr.plot(event.index, event, color='red', marker='o', zorder=2)
    plt.show()
