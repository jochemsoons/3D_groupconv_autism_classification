import numpy as np

array_1 = np.array([1, 2 , 3 ,4, 5, 6, 7, 8])
array_2 = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
array_3 = np.array([11, 22, 33, 44, 55, 66, 77, 88])

p = np.random.permutation(len(array_1))
array_1 = array_1[p]
array_2 = array_2[p]
array_3 = array_3[p]

print(array_1)
print(array_2)
print(array_3)