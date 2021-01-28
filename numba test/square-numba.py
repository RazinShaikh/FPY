# %%
import numpy as np
import time
import cProfile as profile
from neural_network import NNTrain

# %%
a = NNTrain(nx=60, bx=300, hidden_nodes=15, alpha=1e-4, batch_size=50)


# profile.run("a.train(10)", sort="time")

start = time.time()
a.train(10)
print(time.time() - start)
