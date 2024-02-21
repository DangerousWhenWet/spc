from typing import Literal, Optional, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def hex_to_rgba(hex_color:str, a:float):
    if a > 1.0:
        a /= 255.0
    r,g,b = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {a})"


class SPCTrace:
    ZONE_MATPLOTLIB_VISUALIZATION_DEFINITIONS = {
        'a': {'color': 'red', 'linestyle': 'dotted'},
        'b': {'color': 'orange', 'linestyle': 'dotted'},
        'c': {'color': 'yellow', 'linestyle': 'dotted'},
        'center': {'color': 'lime', 'linestyle': 'solid'}
    } # for debug, testing
    RULE_PLOTLY_VISUALIZATION_DEFINITIONS = {
        1: dict(color=hex_to_rgba('#FF0000', 0.667), size=20),
        2: dict(color=hex_to_rgba('#FF7700', 0.500), width=15),
        3: dict(color=hex_to_rgba('#FFFF00', 0.500), width=10),
        4: dict(color=hex_to_rgba('#2A52BD', 0.500), width=7)
    }

    def __init__(self, data:pd.Series, centerline:float, sigma:float):
        self.data = data
        self.centerline = centerline
        self.sigma = sigma

    
    def get_zone(self, zone:Literal['a', 'b', 'c', 'center']):
        multiple = {'a':3, 'b':2, 'c':1, 'center': 0}[zone]
        return (
            self.centerline - multiple * self.sigma,
            self.centerline + multiple * self.sigma
        )

    
    def get_weco_rule1_events(self, condense_recurring_events=False):
        '''any single point beyond beyond ± 3σ'''
        zone_low, zone_high = self.get_zone('a')
        
        def rule1(x):
            return zone_low > x or x > zone_high
            
        hits = self.data.apply(rule1)
        return hits

    @staticmethod
    def _windowed_threshold_count(window, threshold_low, threshold_high, minimum_count):
        '''
        count how many values in the window are above or below given thresholds
        '''
        count_beyond_low = (window < threshold_low).sum()
        count_beyond_high = (window > threshold_high).sum()
        return count_beyond_low >= minimum_count or count_beyond_high >= minimum_count

    @staticmethod
    def _condense_recurring_events(ser):
        '''
        find the place where an event stopped occurring `.diff() == -1.0`, then take 1 position backwards from there
        '''
        return ser.diff().shift(-1) == -1.0
    
    def get_weco_rule2_events(self, condense_recurring_events=False):
        '''2 out of 3 consecutive points beyond ± 2σ in the same direction'''
        zone_low, zone_high = self.get_zone('b')
        
        hits = self.data.rolling(3).apply(
            SPCTrace._windowed_threshold_count,
            args=(zone_low, zone_high, 2)
        ).fillna(0)
        if condense_recurring_events:
            return SPCTrace._condense_recurring_events(hits)
        else:
            return hits.astype(bool)


    def get_weco_rule3_events(self, condense_recurring_events=False):
        '''4 out of 5 consecutive points beyond ± 1σ in the same direction'''
        zone_low, zone_high = self.get_zone('c')
        
        hits = self.data.rolling(5).apply(
            SPCTrace._windowed_threshold_count,
            args=(zone_low, zone_high, 4)
        ).fillna(0)
        if condense_recurring_events:
            cleaned_hits = SPCTrace._condense_recurring_events(hits)
            return cleaned_hits
        else:
            return hits.astype(bool)


    def get_weco_rule4_events(self, condense_recurring_events=False):
        '''8 consecutive points beyond on the same side of centerline'''

        def rule4(window):
            count_beyond_high = (window > self.centerline).sum()
            count_beyond_low = (window < self.centerline).sum()
            return count_beyond_high == 8 or count_beyond_low == 8
            
        hits = self.data.rolling(8).apply(rule4).fillna(0)
        if condense_recurring_events:
            return SPCTrace._condense_recurring_events(hits)
        else:
            return hits.astype(bool)


    def get_weco_event_slices(self, rule_number:int):
        '''generates 1 Series containing all windowed points for each detected event'''
        switcher = {
            1: (self.get_weco_rule1_events, 1),
            2: (self.get_weco_rule2_events, 3),
            3: (self.get_weco_rule3_events, 5),
            4: (self.get_weco_rule4_events, 8),
        }
        rule_finder, lookback = switcher[rule_number]
        events = self.data[rule_finder(condense_recurring_events=True)]

        for event_idx, _ in events.items():
            idx_position = self.data.index.get_loc(event_idx)+1
            event_slice = self.data.iloc[idx_position - lookback: idx_position]
            yield event_slice


def draw_spc_matplotlib(ax, trace:SPCTrace, y_label:Optional[str]=None):
    for zone, definition in SPCTrace.ZONE_MATPLOTLIB_VISUALIZATION_DEFINITIONS.items():
        zone_low, zone_high = trace.get_zone(zone)
        ax.axhline(zone_high, color=definition['color'], linestyle=definition['linestyle'], label=f"Zone {zone.upper()}" if zone in 'abc' else None)
        if zone in 'abc':
            ax.axhline(zone_low, color=definition['color'], linestyle=definition['linestyle'], label=f"Zone {zone.upper()}")
    ax.plot(trace.data.index, trace.data, marker='o', color='black')
    if y_label:
        ax.set_ylabel(y_label)


def draw_spc_plotly(fig:go.Figure, trace:SPCTrace, show_weco_rules:Optional[List[int]]=None, row_number:int=1, column_number:int=1, **kwargs):
    if 2 in show_weco_rules:
        if event_slices := list(trace.get_weco_event_slices(2)):
            points = pd.concat( event_slices )
            points = points[ ~points.index.duplicated(keep='first') ] # duplicate indices when the left and right edges of two back-to-back event slices overlap
            points = points.reindex_like(trace.data)
            fig.add_trace(go.Scatter(x=points.index, y=points, connectgaps=False, showlegend=False, mode='lines', line=SPCTrace.RULE_PLOTLY_VISUALIZATION_DEFINITIONS[2]), row=row_number, col=column_number)
    
    if 3 in show_weco_rules:
        if event_slices := list(trace.get_weco_event_slices(3)):
            points = pd.concat( event_slices )
            points = points[ ~points.index.duplicated(keep='first') ] # duplicate indices when the left and right edges of two back-to-back event slices overlap
            points = points.reindex_like(trace.data)
            fig.add_trace(go.Scatter(x=points.index, y=points, connectgaps=False, showlegend=False, mode='lines', line=SPCTrace.RULE_PLOTLY_VISUALIZATION_DEFINITIONS[3]), row=row_number, col=column_number)
    
    if 4 in show_weco_rules:
        if event_slices := list(trace.get_weco_event_slices(4)):
            points = pd.concat( event_slices )
            points = points[ ~points.index.duplicated(keep='first') ] # duplicate indices when the left and right edges of two back-to-back event slices overlap
            points = points.reindex_like(trace.data)
            fig.add_trace(go.Scatter(x=points.index, y=points, connectgaps=False, showlegend=False, mode='lines', line=SPCTrace.RULE_PLOTLY_VISUALIZATION_DEFINITIONS[4]), row=row_number, col=column_number)
    
    if 1 in show_weco_rules:
        if event_slices := list(trace.get_weco_event_slices(1)):
            points = pd.concat( event_slices )
            points = points[ ~points.index.duplicated(keep='first') ] # duplicate indices when the left and right edges of two back-to-back event slices overlap
            points = points.reindex_like(trace.data)
            fig.add_trace(go.Scatter(x=points.index, y=points, showlegend=False, mode='markers', marker=SPCTrace.RULE_PLOTLY_VISUALIZATION_DEFINITIONS[1]), row=row_number, col=column_number)

    fig.add_trace(go.Scatter(
        x=trace.data.index, y=trace.data, showlegend=False, **kwargs
    ), row=row_number, col=column_number)
    for value, line_dict, label in [
                (trace.centerline, dict(width=3, dash='solid', color='grey'), 'Average'),
                (trace.centerline + 3 * trace.sigma, dict(width=2, dash='dash', color='grey'), 'UCL'),
                (trace.centerline - 3 * trace.sigma, dict(width=2, dash='dash', color='grey'), 'LCL'),
            ]:
        fig.add_hline(y=value, line=line_dict, annotation_text=f"{label}: {value:.03f}", row=row_number, col=column_number)


