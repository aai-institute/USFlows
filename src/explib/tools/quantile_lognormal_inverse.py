import math
from scipy.stats import norm
import numpy


def quantile_log_normal(p, mu=1, sigma=0.5):
    return math.exp(mu + sigma * norm.ppf(p))

def inverse_log_normal(val, mu=1, sigma=0.5):
    return norm.cdf((1/sigma)*(math.log(val)-mu))

def invers_quantile_lognormal():

    for p in [0.5, 0.3, 0.2, 0.1, 0.01, 0.003]:
        print(f'p = {p}, r={quantile_log_normal(p)}')
        print(f'p = {p}, reconstructed={inverse_log_normal(quantile_log_normal(p))}')
    return
    lo = 0
    hi = 0.2
    p = (hi - lo) / 2
    val = quantile_log_normal(p)
    while abs(val - 1.5) >= 0.0001:
        if val < 1.5:
            lo = lo + ((hi-lo)/2)
        if val >= 1.5:
            hi = hi - ((hi-lo)/2)
        p = lo+ (hi-lo)/2
        val = quantile_log_normal(p)
    print(f'p={p} , r={quantile_log_normal(p)}')

if __name__ == '__main__':
    invers_quantile_lognormal()



