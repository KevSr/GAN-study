import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

def Rosenbrock(x, y):
    return (1 + x)**2 + 100*(y-x**2)**2

def GradRosenbrock(x, y):
    g1= -400*x*y + 400*x**3 + 2*x - 2
    g2= 200*y - 200*x**2
    return np.array([g1, g2])

def HessianRosenbrock(x, y):
    h11= -400*y + 1200*x**2 + 2
    h12= -400*x
    h21= -400*x
    h22= 200
    return np.array([[h11, h12], [h21, h22]])

def GradientDescent(Grad, x, y, gamma=0.00125, epsilon= 0.0001, nMax= 10000):
    #initialization
    i=0
    iterX, iterY, iterCount= np.empty(0), np.empty(0), np.empty(0)
    error= 10
    X= np.array([x, y])

    while np.linalg.norm(error) > epsilon and i < nMax:
        i += 1
        iterX= np.append(iterX, x)
        iterY= np.append(iterY, y)
        iterCount= np.append(iterCount, i)

        prevX= X
        X= X- gamma * Grad(x, y)
        error= X- prevX
        x, y= X[0], X[1]

    print(X)
    return X, iterX, iterY, iterCount

def Himmer(x,y):
    return (x**2 + y - 11)**2 + ( x + y**2 - 7 )**2

def GradHimmer(x,y):
    return np.array([2 * (-7 + x + y**2 + 2 * x * (-11 + x**2 + y)), 2 * (-11 + x**2 + y + 2 * y * (-7 + x + y**2))])

def HessianHimmer(x,y):
    h11 = 4 * (x**2 + y - 11) + 8 * x**2 + 2
    h12 = 4 * x + 4 * y
    h21 = 4 * x + 4 * y 
    h22 = 4 * (x + y**2 - 7) + 8 * y**2 + 2
    
    return np.array([[h11,h12],[h21,h22]])

def NewtonRaphsonOptimize(Grad, Hess, x, y, epsilon=0.000001, nMax= 200):
    #initialization
    i= 0
    iterX, iterY, iterCount= np.empty(0), np.empty(0), np.empty(0)
    error= 10
    X= np.array([x, y])

    while np.linalg.norm(error) > epsilon and i < nMax:
        i += 1
        iterX= np.append(iterX, x)
        iterY= np.append(iterY, y)
        iterCount= np.append(iterCount, i)
        print(X)

        prevX= X
        X= X- np.linalg.inv(Hess(x,y)) @ Grad(x, y)
        error= X- prevX
        x, y= X[0], X[1]

    return X, iterX, iterY, iterCount

plt.style.use('seaborn-white')


# x= np.linspace(-3, 3, 250)
# y= np.linspace(-9, 8, 350)
# X, Y= np.meshgrid(x, y)
# Z= Rosenbrock(X, Y)
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = Himmer(X, Y)


#===================================== Gradient descent 
# root_gd,iterX_gd,iterY_gd, iter_count_gd = GradientDescent(GradRosenbrock, -2, 2)
root_gd,iterX_gd,iterY_gd, iter_count_gd = GradientDescent(GradHimmer,0.5, -2)
# root_nr,iterX_nr,iterY_nr, iter_count_nr = NewtonRaphsonOptimize(GradRosenbrock, HessianRosenbrock, -2, 2)
root_nr,iterX_nr,iterY_nr, iter_count_nr = NewtonRaphsonOptimize(GradHimmer,HessianHimmer, 0.5, -2)

anglesx = iterX_gd[1:] - iterX_gd[:-1]
anglesy = iterY_gd[1:] - iterY_gd[:-1]
anglesx_nr = iterX_nr[1:] - iterX_nr[:-1]
anglesy_nr = iterY_nr[1:] - iterY_nr[:-1]

#%matplotlib inline
fig = plt.figure(figsize = (16,8))

#Surface plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
# ax.plot(iterX_gd,iterY_gd, Rosenbrock(iterX_gd,iterY_gd),color = 'r', marker = '*', alpha = .4, label = 'Gradient descent')
# ax.plot(iterX_nr,iterY_nr, Rosenbrock(iterX_nr,iterY_nr),color = 'darkblue', marker = 'o', alpha = .4, label = 'Newton')
ax.plot(iterX_gd,iterY_gd, Himmer(iterX_gd,iterY_gd),color = 'r', marker = '*', alpha = .4, label = 'Gradient descent')
ax.plot(iterX_nr,iterY_nr, Himmer(iterX_nr,iterY_nr),color = 'darkblue', marker = 'o', alpha = .4, label = 'Newton')
ax.legend()

ax.view_init(45, 280)
ax.set_xlabel('x');
ax.set_ylabel('y')

# contour plot
ax= fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 60, cmap = 'jet')

ax.scatter(iterX_gd,iterY_gd,color = 'r', marker = '*', label = 'Gradient descent')
ax.quiver(iterX_gd[:-1], iterY_gd[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3)

ax.scatter(iterX_nr,iterY_nr,color = 'darkblue', marker = 'o',  label = 'Newton method')
ax.quiver(iterX_nr[:-1], iterY_nr[:-1], anglesx_nr, anglesy_nr, scale_units = 'xy', angles = 'xy', scale = 1, color = 'darkblue', alpha = .3)
ax.legend()


ax.set_title('Gradient Descent with {} iterations and Newton Method with {} iterations'.format(len(iter_count_gd), len(iter_count_nr)))

# for i, txt in enumerate(iter_count_gd):
#     if i<10:
#         ax.annotate(txt, (iterX_gd[i*2], iterY_gd[i*2]))

# for i, txt in enumerate(iter_count_nr):
#     if i<10:
#         ax.annotate(txt, (iterX_nr[i], iterY_nr[i]))


# animation
# linHimmer_2d_gd, = ax.plot([],[])
# dotHimmer_2d_gd, = ax.plot([],[], color='r', marker='*', alpha=0.4)
# linHimmer_2d_nr, = ax.plot([],[])
# dotHimmer_2d_nr, = ax.plot([],[], color='r', marker='*', alpha=0.4)

# def graphHimmer2d_gd(i):
#     linHimmer_2d_gd.set_data(iterX_gd[:i], iterY_gd[:i])
#     dotHimmer_2d_gd.set_data(iterX_gd[i], iterY_gd[i])

# def graphHimmer2d_nr(i):
#     linHimmer_2d_nr.set_data(iterX_nr[:i], iterY_nr[:i])
#     dotHimmer_2d_nr.set_data(iterX_nr[i], iterY_nr[i])

# anim1= animation.FuncAnimation(fig, graphHimmer2d_gd, frames= len(iter_count_gd), interval=5)
# anim2= animation.FuncAnimation(fig, graphHimmer2d_nr, frames= len(iter_count_gd), interval=20)
plt.show()
