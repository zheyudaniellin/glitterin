import numpy as np
import timeit

import sys
sys.path.append('..')
import glitterin

# Load Producer
print('Load Producer')

producer = glitterin.user.ScatteringProducer(nndir='/home/zylin/coding/glitterin')
producer.setup(['Cext', 'Cabs', 'Z11', 'Z12', 'N12', 'N22', 'N33', 'N34', 'N44'])

def my_function():
    n_sample = 100
    out = producer(
        np.array([1] * n_sample), 
        np.array([1.7] * n_sample), 
        np.array([0.01] * n_sample), 
        np.linspace(0, 180, 181), 
        2 * np.pi)
    return out

print('Start timeit')

number = 20
setup_code = 'from __main__ import my_function'
execution_time = timeit.timeit('my_function()', setup=setup_code, number=number)
print(f'Execution time: {execution_time} seconds')

avg_time = execution_time / number
print(f'Average time: {avg_time} seconds')
