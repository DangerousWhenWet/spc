from typing import Literal, Optional, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative as qualitative_color_scales
D3 = qualitative_color_scales.D3


def clamp(value:float, minimum=float('-Infinity'), maximum=float('Infinity')):
    return max(minimum, min(value, maximum))

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
        1: dict(line_color=hex_to_rgba(D3[3], 0.75), line_width=2, color=hex_to_rgba('#FFFFFF', 0.0), size=20),
        2: dict(color=hex_to_rgba(D3[1], 0.750), width=12),
        3: dict(color=hex_to_rgba(D3[4], 0.500), width=12),
        4: dict(color=hex_to_rgba(D3[9], 0.500), width=12)
    }

    def __init__(self, data:pd.Series, centerline:float, sigma:float, name:Optional[str]=None):
        # if any(data.index.duplicated()):
        #     raise ValueError(f"{self.__class__.__name__} for {data.name} has duplicates in index.\n{data[data.index.duplicated()]}")
        self.data = data
        self.centerline = centerline
        self.sigma = sigma
        self.name=name

    
    def get_zone(self, zone:Literal['a', 'b', 'c', 'center']):
        multiple = {'a':3, 'b':2, 'c':1, 'center': 0}[zone]
        return (
            self.centerline - multiple * self.sigma,
            self.centerline + multiple * self.sigma
        )

    
    def get_weco_rule1_events(self, condense_recurring_events=False, as_reset_index=False):
        '''any single point beyond beyond ± 3σ'''
        zone_low, zone_high = self.get_zone('a')
        
        def rule1(x):
            return zone_low > x or x > zone_high
            
        hits = self.data.apply(rule1)
        return hits.reset_index(drop=True) if as_reset_index else hits

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
        # timestamps can be ambiguous as indices -- they can be duplicated (rarely) -- treat them differently and use integer location within index instead
        is_datetimeindex = pd.api.types.is_datetime64_any_dtype(self.data.index)
        if is_datetimeindex:
            events = self.data.reset_index(drop=True)[rule_finder(condense_recurring_events=True, as_reset_index=is_datetimeindex)]
        else:
            events = self.data[rule_finder(condense_recurring_events=True)]

        # find integer location of the events within indices, so you can take integer "lookback" slices for multi-point events
        for event_idx, _ in events.items():
            idx_position = self.data.index.get_loc(event_idx) if not is_datetimeindex else event_idx
            # get_loc can do some unexpected things if your index contains duplicates and this `event_idx` happens to be one of the duplicates; be defensive about it (but you should get rid of duplicates before getting to this point...)
            if not isinstance(idx_position, int):
                raise ValueError(f"{self.__class__.__name__} for {self.data.name} has bad index while checking for WECO event {rule_number}. (Does the index have duplicates?)\n{idx_position=}")
            idx_position += 1
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


def draw_spc_plotly(fig:go.Figure, trace:SPCTrace, show_weco_rules:Optional[List[int]]=None, row_number:int=1, column_number:int=1, clamp_control_limits:Optional[Tuple[float]]=None, lsl:Optional[float]=None, usl:Optional[float]=None, **kwargs):
    if 2 in show_weco_rules:
        if event_slices := list(trace.get_weco_event_slices(2)):
            points = pd.concat( event_slices )
            points = points[ ~points.index.duplicated(keep='first') ] # duplicate indices when the left and right edges of two back-to-back event slices overlap
            points = points.reindex_like(trace.data)
            fig.add_trace(
                go.Scatter(x=points.index, y=points, connectgaps=False, showlegend=False, mode='lines', line=SPCTrace.RULE_PLOTLY_VISUALIZATION_DEFINITIONS[2], hoverinfo='none'),
            row=row_number, col=column_number)
    
    if 3 in show_weco_rules:
        if event_slices := list(trace.get_weco_event_slices(3)):
            points = pd.concat( event_slices )
            points = points[ ~points.index.duplicated(keep='first') ] # duplicate indices when the left and right edges of two back-to-back event slices overlap
            points = points.reindex_like(trace.data)
            fig.add_trace(
                go.Scatter(x=points.index, y=points, connectgaps=False, showlegend=False, mode='lines', line=SPCTrace.RULE_PLOTLY_VISUALIZATION_DEFINITIONS[3], hoverinfo='none'), 
            row=row_number, col=column_number)
    
    if 4 in show_weco_rules:
        if event_slices := list(trace.get_weco_event_slices(4)):
            points = pd.concat( event_slices )
            points = points[ ~points.index.duplicated(keep='first') ] # duplicate indices when the left and right edges of two back-to-back event slices overlap
            points = points.reindex_like(trace.data)
            fig.add_trace(go.Scatter(x=points.index, y=points, connectgaps=False, showlegend=False, mode='lines', line=SPCTrace.RULE_PLOTLY_VISUALIZATION_DEFINITIONS[4], hoverinfo='none'),
            row=row_number, col=column_number)
    
    # We always show a subtle version of Rule 1 (see below color masks), but make it more obvious if user asked to plot Rule 1
    if event_slices := list(trace.get_weco_event_slices(1)):
        rule1_points = pd.concat( event_slices )
        rule1_points = rule1_points[ ~rule1_points.index.duplicated(keep='first') ] # duplicate indices when the left and right edges of two back-to-back event slices overlap
        rule1_points = rule1_points.reindex_like(trace.data)
        if 1 in show_weco_rules:
            fig.add_trace(
                go.Scatter(x=rule1_points.index, y=rule1_points, showlegend=False, mode='markers', marker=SPCTrace.RULE_PLOTLY_VISUALIZATION_DEFINITIONS[1], hoverinfo='none'),
            row=row_number, col=column_number)
    else:
        rule1_points = pd.Series([np.nan]).reindex_like(trace.data)

    rule1_marker_colors = rule1_points.notna().replace({True: 'red', False: 'green'})
    fig.add_trace(go.Scatter(
        x=trace.data.index, y=trace.data, showlegend=False, marker_line_color=rule1_marker_colors, marker_line_width=2, **kwargs
    ), row=row_number, col=column_number)
    
    if clamp_control_limits:
        ucl = clamp(trace.centerline + 3 * trace.sigma, *clamp_control_limits)
        lcl = clamp(trace.centerline - 3 * trace.sigma, *clamp_control_limits)
    else:
        ucl = trace.centerline + 3 * trace.sigma
        lcl = trace.centerline - 3 * trace.sigma
    
    print(f"{lsl=}, {usl=}")
    for value, line_dict, label in [
                (trace.centerline, dict(width=3, dash='solid', color='grey'), 'Average'),
                (ucl, dict(width=2, dash='dash', color='grey'), 'UCL'),
                (lcl, dict(width=2, dash='dash', color='grey'), 'LCL'),
            ] \
                + ([(lsl, dict(width=2, dash='dot', color='grey'), 'LSL'),] if not any((lsl is None, np.isneginf(lsl or np.nan))) else []) \
                + ([(usl, dict(width=2, dash='dot', color='grey'), 'USL'),] if not any((usl is None, np.isposinf(usl or np.nan))) else []) :
        fig.add_hline(y=value, line=line_dict, annotation_text=f"{label}: {value:.04g}", row=row_number, col=column_number)


