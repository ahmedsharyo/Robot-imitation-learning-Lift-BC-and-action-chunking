import math


def wilson_interval(successes, trials, z=1.96):
    if trials <= 0: raise ValueError("trials must be positive")
    p = successes / trials
    d = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / d
    half = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / d
    return center-half, center+half
