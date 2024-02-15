from typing import Literal, Optional

import numpy as np
import pandas as pd


class SPCTrace:
    ZONE_VISUALIZATION_DEFINITIONS = {
        'a': {'color': 'red', 'linestyle': 'dotted'},
        'b': {'color': 'orange', 'linestyle': 'dotted'},
        'c': {'color': 'yellow', 'linestyle': 'dotted'},
        'center': {'color': 'lime', 'linestyle': 'solid'}
    } # for debug, testing

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
            return SPCTrace._condense_recurring_events(hits)
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
    for zone, definition in SPCTrace.ZONE_VISUALIZATION_DEFINITIONS.items():
        zone_low, zone_high = trace.get_zone(zone)
        ax.axhline(zone_high, color=definition['color'], linestyle=definition['linestyle'], label=f"Zone {zone.upper()}" if zone in 'abc' else None)
        if zone in 'abc':
            ax.axhline(zone_low, color=definition['color'], linestyle=definition['linestyle'], label=f"Zone {zone.upper()}")
    ax.plot(trace.data.index, trace.data, marker='o', color='black')
    if y_label:
        ax.set_ylabel(y_label)