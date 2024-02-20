import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticks

from . import SPCTrace, draw_spc_matplotlib
import spc.factors as factors


class XMRTraces:
    def __init__(self, data:pd.Series):
        ANTIBIAS_D4 = factors.get_D4(n=2)
        ANTIBIAS_d2 = factors.get_d2(n=2)

        mr = data.diff().abs()
        mr_bar = mr.dropna().mean()
        self.mr = SPCTrace(
            data = mr,
            centerline = mr_bar,
            sigma = (ANTIBIAS_D4 * mr_bar - mr_bar)/3
        )
        self.x = SPCTrace(
            data = data,
            centerline = data.mean(),
            sigma = (3/ANTIBIAS_d2 * mr_bar)/3
        )


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
