import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import pandas as pd


from . import SPCTrace, draw_spc_matplotlib
import factors


class XRTraces:
    def __init__(self, data:pd.DataFrame, data_column:str, groupby_column:str, subgroup_size:int):
        ANTIBIAS_D4 = factors.get_D4(n=subgroup_size)
        ANTIBIAS_A2 = factors.get_A2(n=subgroup_size)
        rational_subgroups = data.groupby(groupby_column)[data_column]
        r = rational_subgroups.max() - rational_subgroups.min()
        r_bar = r.mean()
        self.r = SPCTrace(
            data = r,
            centerline = r_bar,
            sigma = (ANTIBIAS_D4 * r_bar - r_bar)/3
        )
        x_bar = rational_subgroups.mean()
        grand_mean = x_bar.mean()
        self.x = SPCTrace(
            data = x_bar,
            centerline = grand_mean,
            sigma = (ANTIBIAS_A2 * r_bar)/3
        )


if __name__ == '__main__':
    np.random.seed(908)

    # generate random subgroup means between 70-90, random subgroup stddev between 5-30
    num_subgroups = 15
    subgroup_means = np.random.uniform(90, 95, size=num_subgroups)
    subgroup_stddevs = np.random.uniform(1, 5, size=num_subgroups)
    subgroup_size = 5

    # generate dummy data for each subgroup and then combine it all
    data_per_subgroup = []
    for mean, stddev in zip(subgroup_means, subgroup_stddevs):
        subgroup_data = pd.DataFrame({
            'value': np.random.normal(loc=mean, scale=stddev, size=subgroup_size),  # Generating 12 values for each subgroup
            'group': np.repeat(len(data_per_subgroup) + 1, repeats=subgroup_size)  # Assigning group numbers
        })
        data_per_subgroup.append(subgroup_data)
    data = pd.concat(data_per_subgroup, ignore_index=True)
    # with pd.option_context('display.max_rows', None):
    #     print(data)

    xr = XRTraces(data, data_column='value', groupby_column='group', subgroup_size=subgroup_size)

    def generate_xr(xr:XRTraces):
        fig, (ax_x, ax_r) = plt.subplots(nrows=2, ncols=1, sharex=True)
        _ = plt.subplots_adjust(hspace=0)
        draw_spc_matplotlib(ax_x, xr.x, 'x̅')
        draw_spc_matplotlib(ax_r, xr.r, 'R')
        ax_r.xaxis.set_major_locator(mticks.MultipleLocator(5))
        ax_r.xaxis.set_minor_locator(mticks.MultipleLocator(1))
        ax_r.set_ylim([-0.05 * (ax_r.get_ylim()[1] - ax_r.get_ylim()[0]) , ax_r.get_ylim()[1]])

        return ax_x, ax_r
    
    ax_x, ax_r = generate_xr(xr)
    for event in xr.x.get_weco_event_slices(rule_number=2):
        ax_x.plot(event.index, event, color='red', marker='o', zorder=2)
    for event in xr.r.get_weco_event_slices(rule_number=2):
        ax_r.plot(event.index, event, color='red', marker='o', zorder=2)
    plt.show()
