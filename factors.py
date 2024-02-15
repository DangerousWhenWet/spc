import functools
import math

import numpy as np
import scipy


def get_d2(n:int) -> float:
    '''Return Hartley's conversion constant (aka: "d2") control chart factor.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    def func(x):
        ncdf = scipy.stats.norm.cdf(x)
        return 1 - (1 - ncdf)**n - (ncdf)**n
    return scipy.integrate.quad(func, -np.inf, np.inf)[0]


def get_d3(n:int) -> float:
    '''Return d3 control chart factor. This one is really slow, lots of numeric calculus in there. You should cache it.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    d2 = get_d2(n)
    def func(x, y):
        ncdf_x = scipy.stats.norm.cdf(x)
        ncdf_y = scipy.stats.norm.cdf(y)
        return 1 - ncdf_y**n - (1-ncdf_x)**n + (ncdf_y - ncdf_x)**n

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
    def fractional_factorial(x:float):
        factors = [(x-i) for i in range(math.floor(x) + 1) if (x-i) > 0] + [math.sqrt(math.pi)]
        return functools.reduce(lambda l,r: l*r, factors)

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
    return 3 / math.sqrt(n)


def get_A2(n:int) -> float:
    ''' Return A2 control chart factor for averages.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    return 3 / (get_d2(n) * math.sqrt(n))


def get_A3(n:int) -> float:
    ''' Return A3 control chart factor for averages.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    return 3 / (get_c4(n) * math.sqrt(n))


def get_B3(n:int) -> float:
    ''' Return B3 control chart factor for standard deviations.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
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
    return 3 / get_c4(n)


def get_E3(n:int) -> float:
    ''' Return E3 control chart factor for individuals.
        Required:
            n: number of observations per sample
        source: "Mathematical Relations and Tables of Factors of Computing Control Chart Lines."
                    ASTM International, Manual on Presentation of Data and Control Chart Analysis - 8th Edition (2010),
                    Chapter 3, Supplement 3.A.
    '''
    return 3 / get_d2(n)


if __name__ == "__main__":
    print("Patience is a virtue. This is just a test and it isn't a very efficient one.\nThe factors related to ranges are the slowest.")

    import pandas as pd

    df = pd.DataFrame({'n': [i for i in range(2, 25+1)]})
    df['A'] = df['n'].apply(get_A)
    df['A2'] = df['n'].apply(get_A2)
    df['A3'] = df['n'].apply(get_A3)
    df['B3'] = df['n'].apply(get_B3)
    df['B4'] = df['n'].apply(get_B4)
    df['B5'] = df['n'].apply(get_B5)
    df['B6'] = df['n'].apply(get_B6)
    df['D1'] = df['n'].apply(get_D1)
    df['D2'] = df['n'].apply(get_D2)
    df['D3'] = df['n'].apply(get_D3)
    df['D4'] = df['n'].apply(get_D4)
    df['E2'] = df['n'].apply(get_E2)
    df['E3'] = df['n'].apply(get_E3)
    df['d2'] = df['n'].apply(get_d2)
    df['d3'] = df['n'].apply(get_d3)
    df['c4'] = df['n'].apply(get_c4)

    df.set_index(['n'], inplace=True)

    print(df)