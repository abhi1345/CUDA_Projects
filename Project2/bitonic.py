global a
a = [8,5,34,2,5,7,9,90,56,3,2,2,56,7,9,1]

import random
amount = 32
L = range(amount*10)
a = [random.choice(L) for _ in range(amount)]

sorted_a = sorted(a)

print(a)

def bitonic_merge(start_index=0, length=len(a), dir="inc"):
    # if length == 2:
    #     print("Starting merge:", start_index, length, dir)    
    if length == 1:
        return

    midpoint = start_index + length//2

    for i in range(start_index, midpoint):
        a_cur, b_cur = a[i], a[i+(length//2)]

        swap = False

        if dir == "inc" and a_cur > b_cur:
            swap = True
        if dir == "dec" and a_cur < b_cur:
            swap = True

        # print("Before swap=", swap, a)
        
        
        if swap:
            a[i] = b_cur
            a[i+length//2] = a_cur

        # print("After swap=", swap, a)

    bitonic_merge(start_index, length//2, dir)
    bitonic_merge(midpoint, length//2, dir)

    return


def bitonic_sort(start_index, length, dir="inc"):
    # if length == 2:
    #     print("Starting sort:", start_index, length, dir)
    if length == 1:
        return

    midpoint = start_index + length // 2
    # print("a before bitsort", a, dir)
    bitonic_sort(start_index, length//2, dir="inc")
    bitonic_sort(midpoint, length//2, dir="dec")
    # print("a after bitsort", a)

    # print("merging a", a, dir)
    bitonic_merge(start_index, length, dir)

    return

import time
start_time = time.time()
bitonic_sort(0, len(a))
end_time = time.time() - start_time
print(a)
print("is correct?: ", a == sorted_a)
print("Sorted in {} seconds".format(end_time))


print("BREAKER")
a = [4, 5, 2, 1, 7, 3, 6, 0, 9, 12, 15, 10, 8, 11, 14, 13]
print(a)
bitonic_merge(0, len(a))
print(a)