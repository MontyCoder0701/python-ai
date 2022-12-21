import matplotlib
import matplotlib.pyplot as plt
print(matplotlib.__version__)

x = [1, 3, 5]
y = [2, 8, 11]

plt.plot(x, y)
print(plt.rcParams.get("figure.figsize"))
plt.title("Graph")
plt.xlabel("This is x", loc="right")
plt.ylabel("This is y")

# Figure, Axes
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

ax1.plot(x, y)
ax1.set_title("Graph")

ax4.plot(x, y, marker="*", label="Eng")

z = [3, 4, 5]

plt.plot(x, z, label="Math")
plt.legend()

plt.show()
