import time
from math import log2

global a
a = [8,5,34,2,5,7,9,90,56,3,2,2,56,7,9,1]

import random
amount = 16
L = range(amount*5)
a = list(L)
random.shuffle(a)
a = a[:amount]

# a = [49, 68, 28, 31, 29, 16, 41, 12, 56, 57, 70, 6, 72, 26, 52, 63]

sorted_a = sorted(a)

print(a)

def bitonic_merge(a, increasing):
    mp = len(a)//2
    for i in range(mp):
        j = i+mp
        if (a[i] > a[j]) == increasing:
            a[i], a[j] = a[j], a[i]
    return a

def bitonic_looped(a):
    log_len = int(log2(len(a)))
    for level in range(1,log_len+1):
        increasing = True
        for window in range(len(a)//2**(level)):
            for sublevel in reversed(range(level+1)):
                for sub_window in range(2**(level-sublevel)):
                    sub_window_start = window*2**(level) + sub_window*2**sublevel
                    sub_window_end = sub_window_start+2**sublevel
                    a[sub_window_start:sub_window_end] = \
                        bitonic_merge(a[sub_window_start:sub_window_end], increasing)
            increasing = not increasing
    return a

def bitonic2(a):
    log_len = int(log2(len(a)))
    for level in range(1,log_len+1): # level = 1 to log(len)
        for sublevel in reversed(range(level)): # sublevel = level-1 to 0
            for first_index in range(len(a)):
                second_index = first_index ^ 2**sublevel
                if second_index > first_index:
                    if 2**level & first_index == 0:
                        if a[first_index] > a[second_index]:
                            a[second_index], a[first_index] = a[first_index], a[second_index]
                    else:
                        if a[first_index] < a[second_index]:
                            a[second_index], a[first_index] = a[first_index], a[second_index]
    return a
        




start_time = time.time()
a = bitonic2(a)
end_time = time.time() - start_time

print(a)
print("is correct?: ", a == sorted_a)
print(max(a[:len(a)//2]), min(a[len(a)//2:]))
print("Sorted in {} seconds".format(end_time))