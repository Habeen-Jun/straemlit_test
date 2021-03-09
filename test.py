import numpy as np
a = np.random.randn(20, 3)
print(a)

l = [1,2,3,4,5]
v = [6,7,8,9,10]

print([[i[0],i[1]] for i in zip(l,v)])

# for i in zip(l,v):
#     print(i)