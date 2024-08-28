import matplotlib.pyplot as plt
import numpy as np

arr = np.load(r'D:\soft\Desktop\FL\FL\log12\loss\client_loss.npy')

plt.plot(arr)
plt.show()