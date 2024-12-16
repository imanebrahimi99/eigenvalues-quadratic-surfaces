"""
By Ruolin Wang, Sebastian Westerlund, Ruth Risberg and Iman Ebrahimi, 2022-03-10

The three tasks are separated into different functions to put them in different scopes and make them run completely separately. All imports are collected at the top of the file and the three functions are called at the bottom.
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import axes3d
from sympy.solvers import solve
from sympy import Symbol 


def Task1():
    
    # Task 1, Ruolin 
    
    # %%
    A = np.array([[1,1,2],[1,2,1],[2,1,1],[2,2,1]],dtype=float)
    b = np.array([1,-1,1,-1])
    # Minimisation problem has an exact theoretical solution by solving the normal equations
    Ap = np.linalg.inv(A.T@A)@A.T #This matrix gives the solution to the normal exuaations
    x = Ap@b
    r = A@x-b
    # Solution, residual and its norm
    print('Format: [solution, residual, norm of residue]')
    print('Numerical solution of normal equations:', x,r,np.linalg.norm(r))
    # The following vector is the exact solution of the normal equations by hand
    x_cand = np.array([3/5,-7/5,33/35])
    r_cand = A@x_cand-b
    print('Solution by hand of the normal equations:', x_cand,r_cand,np.linalg.norm(r_cand))
    
    
    # %%
    def f(x):
        return np.linalg.norm(A@x-b)
    xm = optimize.fmin(f,np.array([1,1,1]), xtol=0.00001, ftol=0.00001)
    rm = A@xm-b
    # Solution, residual and its norm
    print('Numerical solution with scipy:', xm,rm,np.linalg.norm(rm))
    # As we suspected, even the default selection of parameters provided an optimal solution very close to the pseudo-inverse
    
    # %%
    def plot_iteration(xrange,series, title, format="-"):
        plt.figure(figsize=(16, 8))
        plt.suptitle(title, fontsize=12, fontweight='bold')
        plt.plot(xrange, series, "b"+format)
        plt.xlabel("Values")
        plt.ylabel("position")
        plt.grid(True)
        plt.show()
        
    
    # %%
    pts = 101
    ba = np.linspace(-10,10,pts)
    ra = np.zeros(pts,dtype=float)
    for i in range(pts):
        b_ = np.array([1,ba[i],1,ba[i]])
        ra[i] = np.linalg.norm(A@Ap@b_-b_)
    plot_iteration(ba,ra,'Residual r(a)')
    # Inside the expectation interval we are getting an absolute minimum, when residual is equal to zero
    # In theoretical sense it is possible, but the real result depends on A matrix and vector b
    # It means that we are getting an exact solution.
    # More interesting that in the case of an over-determined system we may have only one, or even infinite family of exact solutions



def Task2():
    # Task 2, Ruth and Sebastian
    finalv = np.array([3, 1, 3])/np.sqrt(19)
    finalq = 4
    
    
    def nextval(z): # If z_n is to be calculated for some n > about 10, the numbers will get too big for numpy arrays but this approach still works.
        a = z[0] + 3*z[1] + 2*z[2]
        b = -3*z[0] + 4*z[1] + 3*z[2]
        c = 2*z[0] + 3*z[1] + z[2]
        return([a, b, c])
    
    def findi(epsilon, findv): # findv toggles whether to search for v or q
        z = [8, 3, 12]
        A = np.array([[1, 3, 2],
                      [-3, 4, 3],
                      [2, 3, 1]])
        v = np.array(z)
        q = v.T@A@v
        
        for i in range(10000): # the for-loop is used to avoid an infinite loop
            # By running this loop with the next few lines uncommented, we can see that all parts of z seem to tend towards infinity, while v and q seem to converge.
            #z = nextval(z)
            #print('z': z)
            #print('v': v)
            #print('q': q)

            
            v = A@v # A is multiplied by v directly since z becomes too big
            v = v/np.linalg.norm(v)
            q = v.T@A@v
            if np.linalg.norm(v-finalv) < epsilon and findv:
                return(i)
            if np.linalg.norm(q-finalq) < epsilon and (not findv):
                return(i)
        return(-1)
    
    print("Number of iterates for ||v_n-v|| < 10^(-8): " + str(findi(10**-8, True)))

    # Calculate data for the plot
    epsilons = [10**i for i in range(-1, -15, -1)]
    vplot = []
    qplot = []
    for epsilon in epsilons:
        vplot.append(findi(epsilon, True))
        qplot.append(findi(epsilon, False))

    # Plotting and formatting
    plt.figure()
    plt.plot(epsilons, vplot, 'b.', label='v')
    plt.plot(epsilons, qplot, 'g.', label='q')
    plt.ylabel('Number of iterations')
    plt.xlabel('epsilon')
    plt.semilogx()
    plt.legend()
    plt.show()
    
    # Answers to the questions in task 2:
    # z_n does not seem to converge
    # v_n seems to converge to about [0.6882472, 0.22941573, 0.6882472]. The exact value of v is [3, 1, 3]/sqrt(19)
    # z and v would be eigenvectors of A, since multiplying them by A would have the same result as multiplying them by the scalar 1.
    # q_n seems to converge towards 4
    # q would be the eigenvalue associated with v, which when calculated is exactly 4.
    # 56 iterates are needed to satisfy ||v_n-v||<10^(-8)
    # v always seems to converge slightly faster than q
    


def Task3():
    # Task3, Iman
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    
    polynomial = 2*x1**2 - x2**2 + 2*x3**2 - 10*x1*x2 -4*x1*x3 + 10*x2*x3
    
    #Finding the solutions of x3 in dependence of x1 and x2
    x3 = solve(polynomial - 1, x3)
    print(f'The Solutions of x3 in dependence of x1 and x2 = \n {x3}\n\n')
    leftHand = x3[0]
    print(f'lower-part = {leftHand}')
    rightHand = x3[1]
    print(f'upper-part = {rightHand}\n\n')
    
    
    
    #Plotting The Figure
    fig = plt.figure()
    ax = fig.gca(projection='3d') #Gives us 3D Axes x,y,z
    X1 = np.linspace(-1,1)
    X2 = np.linspace(-1,1)
    
    #creating a rectangular grid. using an array of X1 and X2 Values
    X1, X2 = np.meshgrid(X1, X2) 
    
    X3 = X1 - 5*X2/2 - np.sqrt(27*X2**2 + 2)/2
    ax.plot_surface(X1, X2, X3) #Lower Part (Blue)
    
    X3 = X1 - 5*X2/2 + np.sqrt(27*X2**2 + 2)/2
    ax.plot_surface(X1, X2, X3) #Upper Part (Orange)
    
    


# Call functions for each task
print("######## Task 1 ########")
Task1()
print("######## Task 2 ########")
Task2()
print("######## Task 3 ########")
Task3()

