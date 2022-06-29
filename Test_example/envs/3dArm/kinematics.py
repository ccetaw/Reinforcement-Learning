import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from sympy import pprint, init_printing
import matplotlib.pyplot as plt

init_printing(wrap_line=False)
theta1, theta2, theta3, theta4, theta5, l1, l2, theta, alpha, r, d, pi = dynamicsymbols(
    'theta1 theta2 theta3 theta4 theta5 l1 l2 theta alpha r d pi')

# Create rotation
rot = sp.Matrix([[sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha)],
                 [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha)],
                 [0, sp.sin(alpha), sp.cos(alpha)]])
# translation
trans = sp.Matrix([r * sp.cos(theta), r * sp.sin(theta), d])

last_row = sp.Matrix([[0, 0, 0, 1]])
T = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

# create kineamtic chain
##some random kinematic chain. Where two joints are in the same postion but rotated from eachother.
m01 = T.subs({alpha: pi / 2, r: 0, theta: theta1, d: 0})
m12 = T.subs({alpha: 0, r: 0, theta: theta2 - pi / 2, d: 0})
m23 = T.subs({alpha: 0, r: l1, theta: theta3 -pi /2, d: 0})
m34 = T.subs({alpha: 0, r: l2, theta: theta4, d: 0})

m02 = (m01 * m12 * m23 * m34)
mbee = sp.Matrix([[m02[0, 0].simplify(), m02[0, 1].simplify(), sp.trigsimp(m02[0, 3].simplify())],
                  [m02[1, 0].simplify(), m02[1, 1].simplify(), sp.trigsimp(m02[1, 3].simplify())],
                  [m02[2, 0].simplify(), m02[2, 1].simplify(), sp.trigsimp(m02[2, 3].simplify())]])
px = mbee[0, 2]
py = mbee[1, 2]
pz = mbee[2, 2]

fx = sp.lambdify((l1, l2, theta1, theta2, theta3, theta4, pi), px, 'numpy')
fy = sp.lambdify((l1, l2, theta1, theta2, theta3, theta4, pi), py, 'numpy')
fz = sp.lambdify((l1, l2, theta1, theta2, theta3, theta4, pi), pz, 'numpy')

l1 = 0.5  # m
l2 = 0.5
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
t1 = 0
t2 = 0
t3 = 0
t4 = 0
xs = []
ys = []
zs = []
for t1 in np.linspace(0, np.pi / 2):
    x = fx(l1, l2, t1, t2, t3, t4, np.pi)
    y = fy(l1, l2, t1, t2, t3, t4, np.pi)
    z = fz(l1, l2, t1, t2, t3, t4, np.pi)
    xs.append(x)
    ys.append(y)
    zs.append(z)

ax.scatter(xs, ys, zs, c='red')
ax.plot(xs, ys, zs, color='red')

xs = []
ys = []
zs = []
for t2 in np.linspace(0, np.pi / 2):
    x = fx(l1, l2, t1, t2, t3, t4, np.pi)
    y = fy(l1, l2, t1, t2, t3, t4, np.pi)
    z = fz(l1, l2, t1, t2, t3, t4, np.pi)
    xs.append(x)
    ys.append(y)
    zs.append(z)

ax.scatter(xs, ys, zs, c='blue')
ax.plot(xs, ys, zs, color='blue')

xs = []
ys = []
zs = []
for t3 in np.linspace(0, np.pi / 2):
    x = fx(l1, l2, t1, t2, t3, t4, np.pi)
    y = fy(l1, l2, t1, t2, t3, t4, np.pi)
    z = fz(l1, l2, t1, t2, t3, t4, np.pi)
    xs.append(x)
    ys.append(y)
    zs.append(z)

ax.scatter(xs, ys, zs, c='green')
ax.plot(xs, ys, zs, color='green')

xs = []
ys = []
zs = []

for t4 in np.linspace(0, np.pi / 2):
    x = fx(l1, l2, t1, t2, t3, t4, np.pi)
    y = fy(l1, l2, t1, t2, t3, t4, np.pi)
    z = fz(l1, l2, t1, t2, t3, t4, np.pi)
    xs.append(x)
    ys.append(y)
    zs.append(z)
ax.scatter(xs, ys, zs, c='yellow')
ax.plot(xs, ys, zs, color='yellow')

xs = []
ys = []
zs = []
for t3 in np.linspace(np.pi / 2, 0):
    x = fx(l1, l2, t1, t2, t3, t4, np.pi)
    y = fy(l1, l2, t1, t2, t3, t4, np.pi)
    z = fz(l1, l2, t1, t2, t3, t4, np.pi)
    xs.append(x)
    ys.append(y)
    zs.append(z)

for t2 in np.linspace(np.pi / 2, 0):
    x = fx(l1, l2, t1, t2, t3, t4, np.pi)
    y = fy(l1, l2, t1, t2, t3, t4, np.pi)
    z = fz(l1, l2, t1, t2, t3, t4, np.pi)
    xs.append(x)
    ys.append(y)
    zs.append(z)
for t1 in np.linspace(np.pi / 2, 0):
    x = fx(l1, l2, t1, t2, t3, t4, np.pi)
    y = fy(l1, l2, t1, t2, t3, t4, np.pi)
    z = fz(l1, l2, t1, t2, t3, t4, np.pi)
    xs.append(x)
    ys.append(y)
    zs.append(z)
for t4 in np.linspace(np.pi / 2, 0):
    x = fx(l1, l2, t1, t2, t3, t4, np.pi)
    y = fy(l1, l2, t1, t2, t3, t4, np.pi)
    z = fz(l1, l2, t1, t2, t3, t4, np.pi)
    xs.append(x)
    ys.append(y)
    zs.append(z)

ax.scatter(xs, ys, zs, c='purple')
ax.plot(xs, ys, zs, color='purple')


ax.scatter([0], [0], [0], c='black', s=200)
plt.show()
