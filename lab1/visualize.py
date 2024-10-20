import numpy as np
import matplotlib.pyplot as plt

N = 50

x = np.fromfile("build/myVec.bin", dtype=np.float32).reshape((N, N))
b = np.fromfile("build/vecB.bin", dtype=np.float32).reshape((N, N))

a = np.fromfile("build/matA.bin", dtype=np.float32).reshape((N*N, N*N))

fig, (ax1, ax2) = plt.subplots(ncols=2)

ax1.imshow(x)
ax1.set_title('Vector X')

ax2.imshow(-b)
ax2.set_title('Vector B')

plt.savefig("myVersion.png")
