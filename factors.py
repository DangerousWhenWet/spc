import concurrent.futures as cf
from dataclasses import dataclass
import functools
import math
import pathlib as pl
from typing import Callable, Dict, Optional
import warnings

import pandas as pd
import numpy as np
import scipy


class ComputationWarning(Warning): pass


LOOKUP_TABLE_PATH = pl.Path(__file__).parent / 'spc_factors_lookup_table.csv'
LOOKUP_TABLE = pd.read_csv(LOOKUP_TABLE_PATH).set_index(['symbol', 'n']) if LOOKUP_TABLE_PATH.exists() else None


def get_d2(n:int) -> float:
    '''Return Hartley's conversion constant (aka: "d2") control chart factor.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    def func(x):
        ncdf = scipy.stats.norm.cdf(x)
        return 1 - (1 - ncdf)**n - (ncdf)**n
    
    if cached_value := fetch_from_lookup_table('d2', n):
        return cached_value
    else:
        return scipy.integrate.quad(func, -np.inf, np.inf)[0]


def get_d3(n:int) -> float:
    '''Return d3 control chart factor. This one is really slow, lots of numeric calculus in there. You should cache it.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')
    d2 = get_d2(n)

    def func(x, y):
        ncdf_x = scipy.stats.norm.cdf(x)
        ncdf_y = scipy.stats.norm.cdf(y)
        return 1 - ncdf_y**n - (1-ncdf_x)**n + (ncdf_y - ncdf_x)**n

    if cached_value := fetch_from_lookup_table('d3', n):
        return cached_value
    else:
        y_lower, y_upper = -np.inf, np.inf
        x_lower, x_upper= lambda y: -np.inf, lambda y: y
        integral = scipy.integrate.dblquad(func, y_lower, y_upper, x_lower, x_upper)[0]
        return math.sqrt(2*integral - d2**2)


def get_c4(n:int) -> float:
    ''' Return c4 control chart factor.
        Required:
            n: number of observations per sample
        source: NIST Engineering Statistics Handbook: Section 6.3.2. What are Variables Control Charts?
                    Accessed April 21, 2023, from NIST/SEMATECH e-Handbook of Statistical Methods
                    website: https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc321.htm.
    '''
    if n < 2: return float('nan')

    def fractional_factorial(x:float):
        factors = [(x-i) for i in range(math.floor(x) + 1) if (x-i) > 0] + [math.sqrt(math.pi)]
        return functools.reduce(lambda l,r: l*r, factors)

    if cached_value := fetch_from_lookup_table('c4', n):
        return cached_value
    else:
        a = math.sqrt(2/(n-1))
        b = math.factorial(int(n/2-1)) if not n%2 else fractional_factorial(n/2-1)
        c = math.factorial(int((n-1)/2-1)) if not (n-1)%2 else fractional_factorial((n-1)/2-1)
        return a * b / c


def get_A(n:int) -> float:
    ''' Return A control chart factor for averages.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    if cached_value := fetch_from_lookup_table('A', n):
        return cached_value
    else:
        return 3 / math.sqrt(n)


def get_A2(n:int) -> float:
    ''' Return A2 control chart factor for averages.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    if cached_value := fetch_from_lookup_table('A2', n):
        return cached_value
    else:
        return 3 / (get_d2(n) * math.sqrt(n))


def get_A3(n:int) -> float:
    ''' Return A3 control chart factor for averages.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    if cached_value := fetch_from_lookup_table('A3', n):
        return cached_value
    else:
        return 3 / (get_c4(n) * math.sqrt(n))


def get_B3(n:int) -> float:
    ''' Return B3 control chart factor for standard deviations.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    if cached_value := fetch_from_lookup_table('c4', n):
        return cached_value
    else:
        c4 = get_c4(n)
        return max(0.0, 1 - (3/c4) * math.sqrt(1 - c4**2))


def get_B4(n:int) -> float:
    ''' Return B4 control chart factor for standard deviations.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    if cached_value := fetch_from_lookup_table('B4', n):
        return cached_value
    else:
        c4 = get_c4(n)
        return 1 + (3/c4) * math.sqrt(1 - c4**2)


def get_B5(n:int) -> float:
    ''' Return B5 control chart factor for standard deviations.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    if cached_value := fetch_from_lookup_table('B5', n):
        return cached_value
    else:
        c4 = get_c4(n)
        return max(0.0, c4 - 3 * math.sqrt(1 - c4**2))


def get_B6(n:int) -> float:
    ''' Return B6 control chart factor for standard deviations.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    if cached_value := fetch_from_lookup_table('B6', n):
        return cached_value
    else:
        c4 = get_c4(n)
        return c4 + 3 * math.sqrt(1 - c4**2)


def get_D1(n:int) -> float:
    ''' Return D1 control chart factor for ranges. This one is really slow, lots of numeric calculus in there. You should cache it.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    if cached_value := fetch_from_lookup_table('D1', n):
        return cached_value
    else:
        if n <= 6: return 0 # silly heuristic save lot of computation; at/below 6 this evaluates to a negative number which is irrelevant
        d2 = get_d2(n)
        d3 = get_d3(n)
        return d2 - 3*d3


def get_D2(n:int) -> float:
    ''' Return D2 control chart factor for ranges. This one is really slow, lots of numeric calculus in there. You should cache it.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    if cached_value := fetch_from_lookup_table('D2', n):
        return cached_value
    else:
        d2 = get_d2(n)
        d3 = get_d3(n)
        return d2 + 3*d3


def get_D3(n:int) -> float:
    ''' Return D3 control chart factor for ranges. This one is really slow, lots of numeric calculus in there. You should cache it.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    if cached_value := fetch_from_lookup_table('D3', n):
        return cached_value
    else:
        if n <= 6: return 0 # silly heuristic save lot of computation; at/below 6 this evaluates to a negative number which is irrelevant
        d2 = get_d2(n)
        d3 = get_d3(n)
        return 1 - 3 * (d3/d2)


def get_D4(n:int) -> float:
    ''' Return D4 control chart factor for ranges. This one is really slow, lots of numeric calculus in there. You should cache it.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    if cached_value := fetch_from_lookup_table('D4', n):
        return cached_value
    else:
        d2 = get_d2(n)
        d3 = get_d3(n)
        return 1 + 3 * (d3/d2)


def get_E2(n:int) -> float:
    ''' Return E2 control chart factor for individuals.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')

    if cached_value := fetch_from_lookup_table('E2', n):
        return cached_value
    else:
        return 3 / get_c4(n)


def get_E3(n:int) -> float:
    ''' Return E3 control chart factor for individuals.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    if n < 2: return float('nan')
    
    if cached_value := fetch_from_lookup_table('E3', n):
        return cached_value
    else:
        return 3 / get_d2(n)




@dataclass
class Task:
    symbol: str
    n: int
    function: Callable
    result: Optional[float] = None


def generate_lookup_table(n_lower:int=2, n_upper:int=200) -> None:
    '''
    Use ProcessPoolExecutor to quickly calculate bulk SPC antibias factors (which are computationally expensive).
    Cache results as a CSV flatfile that the factor getter functions can consult to avoid calculating again.
    '''

    map_symbol_to_getter = {
        'A': get_A,
        'A2': get_A2,
        'A3': get_A3,
        'B3': get_B3,
        'B4': get_B4,
        'B5': get_B5,
        'B6': get_B6,
        'D1': get_D1,
        'D2': get_D2,
        'D3': get_D3,
        'D4': get_D4,
        'E2': get_E2,
        'E3': get_E3,
        'd2': get_d2,
        'd3': get_d3,
        'c4': get_c4
    }
    
    # Start by enqueuing the pool of commands to process, remembering which combination of inputs ("Task") generated each result ("future")
    map_future_to_task: Dict[Task, cf.Future] = {}
    with cf.ProcessPoolExecutor() as executor:
        for n in range(n_lower, n_upper+1):
            for symbol, function in map_symbol_to_getter.items():
                task = Task(symbol=symbol, n=n, function=function)
                future = executor.submit(task.function, task.n)
                map_future_to_task[future] = task

    # Await completion of processing, then extract results into a DataFrame
    records = []
    for completed_future in cf.as_completed(map_future_to_task):
        task = map_future_to_task[completed_future]
        task.result = completed_future.result()
        records.append( dict(n=task.n, symbol=task.symbol, value=task.result) )
    df = pd.DataFrame.from_records(records)

    # Cleanup and save
    df['n'] = df['n'].astype(int)
    df = df.set_index(['symbol', 'n']).sort_index()
    df.to_csv(LOOKUP_TABLE_PATH)


def fetch_from_lookup_table(symbol:str, n:int) -> Optional[float]:
    warnings.simplefilter('ignore')
    if LOOKUP_TABLE is not None:
        try:
            return LOOKUP_TABLE.loc[(symbol, n), 'value']
        except KeyError:
            warnings.warn(f"The requested value {symbol}(n={n}) has not been cached in the lookup table. Calculating on-the-fly, which may be very slow. (You might want to use `generate_lookup_table` with a wider range of n values.)", ComputationWarning)
    else:
        warnings.warn("Lookup table has not been computed. Calculating all factors on-the-fly, which may be very slow. (Use `generate_lookup_table`.)", ComputationWarning)
    return None


if __name__ == "__main__":
    generate_lookup_table(n_lower=2, n_upper=200)
    #print(f"{get_d2(25)=}")
