import matplotlib.pyplot as plt
import numpy as np

# Data
stations = ["AJAC", "ANK2", "ALIC", "MIZU"]
lats = np.array([36.4, 39.9, -23.7, 44.0])
A_vals = np.array([1398, 1224, 1321, 0])

plt.figure()

# Scatter plot
plt.scatter(lats, A_vals)

# Simple labels
for lat, A, name in zip(lats, A_vals, stations):
    plt.text(lat, A, name)

plt.xlabel("Latitude (deg)")
plt.ylabel("A value")
plt.title("Global Time-Field Gradient Map")

plt.show()
