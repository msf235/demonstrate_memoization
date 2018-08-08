import numpy as np

randn = np.random.randn
eig = np.linalg.eig


# %% Define function  -- you can think of this as your ''simulator'' if you want
def get_lam_max(n):
    np.random.seed(0)
    X = randn(n, n)
    lams, ev = eig(X)
    lams = np.sort(lams)[::-1]
    return np.real(lams[0])


# %% test
get_lam_max(1000)
get_lam_max(1000)

# %% Implement memoization "by hand"
cache = {}


def get_lam_max_mem1(n):
    n_str = str(n)
    if n_str in cache:
        return cache[n_str]
    else:
        np.random.seed(0)
        X = np.random.randn(n, n)
        lams, ev = np.linalg.eig(X)
        lams = np.sort(lams)[::-1]
        lam_max = np.real(lams[0])
        cache[n_str] = lam_max
        return cache[n_str]


# %% test
get_lam_max_mem1(1000)
get_lam_max_mem1(1000)
print(cache)

get_lam_max_mem1(800)
get_lam_max_mem1(800)
print(cache)


# %% Write a wrapper that implements memoization automatically
def memoize(f):
    cache = {}

    def get_f_return_value(x):
        if x not in cache:
            cache[x] = f(x)
        return cache[x]

    return get_f_return_value


# %% test
get_lam_max_mem2 = memoize(get_lam_max)
get_lam_max_mem2(1000)
get_lam_max_mem2(1000)


# %% show off decorator syntax
@memoize
def get_lam_max_mem3(n):
    np.random.seed(0)
    X = randn(n, n)
    lams, ev = eig(X)
    lams = np.sort(lams)[::-1]
    return np.real(lams[0])


# %% test
get_lam_max_mem3(1000)
get_lam_max_mem3(1000)

# %% Implement memoization to disk "by hand"
import json

file_name = "function_cache.json"


# memo = {}
def get_lam_max_mem4(n):
    n_str = str(n)
    # Open cache
    try:
        cache = json.load(open(file_name, 'r'))
    except (IOError, ValueError):
        cache = {}

    if n_str in cache:
        return cache[n_str]
    else:
        np.random.seed(0)
        X = np.random.randn(n, n)
        lams, ev = np.linalg.eig(X)
        lams = np.sort(lams)[::-1]
        lam_max = np.real(lams[0])

        # update cache and save to disk
        cache[n_str] = lam_max
        json.dump(cache, open(file_name, 'w'))

        return cache[n_str]


# %% test
get_lam_max_mem4(1000)
get_lam_max_mem4(1000)


# %% Implement memoization to disk automatically
def persist_to_file(file_name):
    def decorator(original_func):

        try:
            cache = json.load(open(file_name, 'r'))
        except (IOError, ValueError):
            cache = {}

        def new_func(param):
            if param not in cache:
                cache[param] = original_func(param)
                json.dump(cache, open(file_name, 'w'))
            return cache[param]

        return new_func

    return decorator


@persist_to_file(file_name)
def get_lam_max_mem5(n):
    np.random.seed(0)
    X = randn(n, n)
    lams, ev = eig(X)
    lams = np.sort(lams)[::-1]
    return np.real(lams[0])


# %% Test
get_lam_max_mem5(1000)
get_lam_max_mem5(1000)

# %% Use the joblib library
from joblib import Memory

memory = Memory(cachedir='function_cache_auto')


@memory.cache
def get_lam_max_mem6(n):
    np.random.seed(0)
    X = randn(n, n)
    lams, ev = eig(X)
    lams = np.sort(lams)[::-1]
    return np.real(lams[0])


# %% test
get_lam_max_mem6(1000)
get_lam_max_mem6(1000)
