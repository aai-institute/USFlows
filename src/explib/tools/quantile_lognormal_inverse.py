import math
from scipy.stats import norm
import numpy


def quantile_log_normal(p, mu=1, sigma=0.5):
    return math.exp(mu + sigma * norm.ppf(p))

def inverse_log_normal(val):
    return norm.cdf(2*math.log(val)-2)
    # return norm.ppf(math.log(val))

def invers_quantile_lognormal():

    for p in [0.3, 0.2, 0.1, 0.01, 0.003]:
        print(f'p = {p}, r={quantile_log_normal(p)}')
        # print(f'p = {p}, r={inverse_log_normal(quantile_log_normal(p))}')

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
    target = 0
    PATH = "./classifications.npy"
    arr = numpy.load(PATH)
    arr = arr[0]
    print(arr)
    print(f'max at position {numpy.argmax(arr)}')
    print(f'confidence: {(len(arr)*arr[target] - (numpy.sum(arr)-arr[target]))/len(arr)}')




