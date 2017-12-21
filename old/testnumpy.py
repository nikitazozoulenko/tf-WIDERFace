import numpy as np


x = np.arange(13*2).reshape(13,2)
y = np.array([y for y in x if y[1] < 6])
print(y)


boxes = np.arange(13*4).reshape(13,4)
indices = np.array([0,1,2,3])

print(boxes)
print(boxes[indices])


b = np.arange(13*4).reshape(13,4)
s = np.arange(13*1).reshape(13,1)

final = np.concatenate((b,s), axis=1)
print(final)
print(final.shape)
