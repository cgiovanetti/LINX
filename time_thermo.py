import jax.numpy as jnp
import jax
import sys

sys.path.append("../")
from linx.background import BackgroundModel


import time


default = BackgroundModel()

for i in range(5):
    start = time.time()
    (a_vec) = jax.block_until_ready(default() )
    
    print(time.time() - start)
    