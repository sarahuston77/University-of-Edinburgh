# This is the code used to create the figures in Section 1.4.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

def plot_trapz():
    '''
    Trapezoid rule plot
    '''

    # Trapezoid rule
    def f(x):
        return np.exp(x - 1) * np.sin(2*x - 1.2) + 1.5

    # Create an x-axis with 100 points and estimate the function
    xmin, xmax = -1.2, 1.2
    x_plot = np.linspace(xmin, xmax, 100)
    f_plot = f(x_plot)

    # Create the nodes
    N = 2
    a, b = -1, 1
    x_node = np.linspace(a, b, N)
    f_node = f(x_node)

    # Plot the function
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_plot, f_plot, 'k-', linewidth=2)

    # Plot the trapezoid
    verts = [[x_node[0], 0], [x_node[1], 0],
             [x_node[1], f_node[1]], [x_node[0], f_node[0]]]
    trapz = patches.Polygon(verts, edgecolor='k', facecolor=[0.2, 0.8, 0.9])
    ax.add_patch(trapz)

    # Plot the nodes
    ax.plot(x_node, f_node, 'rx')

    # Label the plots
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    ax.set_title('Trapezoid rule')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0, 2.8])

    plt.show()


def plot_midpoint():
    '''
    Midpoint rule plot
    '''

    # Midpoint rule
    def f(x):
        return np.exp(x - 1) * np.sin(2*x + 1.2) + 1.5

    # Create an x-axis with 100 points and estimate the function
    xmin, xmax = -1.2, 1.2
    x_plot = np.linspace(xmin, xmax, 100)
    f_plot = f(x_plot)

    # Create the nodes
    N = 1
    a, b = -1, 1
    x_node = 0
    f_node = f(x_node)

    # Plot the function
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_plot, f_plot, 'k-', linewidth=2)

    # Plot the rectangle
    verts = [[a, 0], [b, 0],
             [b, f_node], [a, f_node]]
    rect = patches.Polygon(verts, edgecolor='k', facecolor=[0.2, 0.8, 0.9])
    ax.add_patch(rect)

    # Plot the nodes
    ax.plot(x_node, f_node, 'rx')

    # Label the plots
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    ax.set_title('Midpoint rule')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0, 2.8])

    plt.show()


def plot_simpson(M=1):
    '''
    Simpson's rule plot, M sub-intervals
    '''
    # Function to integrate
    def f(x):
        return np.exp(x - 1) * np.sin(2*x - 1.2) + 1.5

    # Create an x-axis with 100 points and estimate the function
    xmin, xmax = -1.2, 1.2
    x_plot = np.linspace(xmin, xmax, 500)
    f_plot = f(x_plot)

    # Draw a figure for the composite rule
    N = 3
    h = 2/M

    # Plot the function
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_plot, f_plot, 'k-', linewidth=2)
    line_styles = ['-.', ':', '--']
    colours = [[0.3, 0.9, 1.0], [0.15, 0.75, 0.85]]

    # Draw each sub-interval
    for i in range(M):
        #  a, b = -1, 1
        a = -1 + i*h
        b = a + h
        x_node = np.linspace(a, b, N)
        f_node = f(x_node)

        # Plot the area
        coeffs = np.polyfit(x_node, f_node, N-1)
        p_plot = np.polyval(coeffs, x_plot)
        #  ax.plot(x_plot, p_plot, 'r-')
        #  ax.fill_between(x_plot, p_plot, where=(abs(x_plot) <= 1), color=[0.2, 0.8, 0.9])
        ax.plot(x_plot, p_plot, color=[0.2, 0.5, 0.7], linestyle=line_styles[i%3])
        ax.fill_between(x_plot, p_plot, where=np.logical_and(x_plot >= a, x_plot <= b), color=colours[i%2])

        # Plot the nodes
        #  ax.plot(x_node, f_node, 'rx', markersize=10)
        ax.plot(x_node, f_node, 'x', color=[0.2, 0.5, 0.7], markersize=10)

    # Label the plots
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    ax.set_title(f'Composite Simpson\'s rule ({M} partitions)')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0, 2.8])

    plt.show()


if __name__ == "__main__":
    plot_trapz()
    plot_midpoint()
    plot_simpson(M=2)
