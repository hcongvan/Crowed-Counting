import tensorflow as tf
import numpy as np
from fcrn import FCRN

inp = np.random.rand(1,576,576,1)
model = FCRN((1,576,576,1))
y = model(inp)
print('test')