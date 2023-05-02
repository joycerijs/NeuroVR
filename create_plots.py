# fit a fifth degree polynomial to the economic data
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


def objective(x, a, b, c):
    return a*(x-b)**2+c


# values accuracy
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180]
y = [0, 0.71, 0.75, 0.8, 0.86, 0.9, 0.94, 0.97, 0.97, 0.97, 0.97, 0.97, 0.96, 0.97, 0.97, 0.97, 0.94, 0.93, 0.91, 0.93, 0.89, 0.90, 0.92]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.grid()
ax.set_xlabel("Eerste x-aantal seconden")
ax.set_ylabel("Score")
ax.set_title('Model score na x-aantal seconden')
ax.plot(x, y, 'o-', color="r", label="Accuracy")

plt.show()
# # curve fit
# popt, _ = curve_fit(objective, x, y)
# # summarize the parameter values
# a, b, c = popt
# print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
# # plot input vs output
# pyplot.scatter(x, y)
# # define a sequence of inputs between the smallest and largest known inputs
# x_line = arange(min(x), max(x), 1)
# # calculate the output for the range
# y_line = objective(x_line, a, b, c)
# # create a line plot for the mapping function
# pyplot.plot(x_line, y_line, '--', color='red')
# pyplot.show()





