import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from sympy import pprint, init_printing
from envs.arms.viewer import ArmViewer
from gym import spaces

init_printing(wrap_line=False)


class Arm:
    def __init__(self):
        self.x_wrist, self.y_wrist, self.z_wrist, self.x_elbow, self.y_elbow, self.z_elbow = self._setup_chain()
        self.position = np.zeros((3, 3))
        self.theta = np.random.normal(0, np.pi, size=4)
        self.angular_velocity = np.zeros(4)

        self.length = np.asarray([0.5, 0.5])
        self.mass = np.asarray([1.5, 1.5])
        self.dt = 0.01
        self.position[1, 0] = self.x_wrist(self.length[0], self.length[1], self.theta[0], self.theta[1], self.theta[2],
                                           self.theta[3], np.pi)
        self.position[1, 1] = self.y_wrist(self.length[0], self.length[1], self.theta[0], self.theta[1], self.theta[2],
                                           self.theta[3], np.pi)
        self.position[1, 2] = self.z_wrist(self.length[0], self.length[1], self.theta[0], self.theta[1], self.theta[2],
                                           self.theta[3], np.pi)
        self.position[2, 0] = self.x_elbow(self.length[0], self.length[1], self.theta[0], self.theta[1], self.theta[2],
                                           np.pi)
        self.position[2, 1] = self.y_elbow(self.length[0], self.length[1], self.theta[0], self.theta[1], self.theta[2],
                                           np.pi)
        self.position[2, 2] = self.z_elbow(self.length[0], self.length[1], self.theta[0], self.theta[1], self.theta[2],
                                           np.pi)

    def _setup_chain(self):
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
        m23 = T.subs({alpha: 0, r: l1, theta: theta3 - pi / 2, d: 0})
        m34 = T.subs({alpha: 0, r: l2, theta: theta4, d: 0})

        m04 = (m01 * m12 * m23 * m34)

        px = sp.trigsimp(m04[0, 3].simplify())
        py = sp.trigsimp(m04[1, 3].simplify())
        pz = sp.trigsimp(m04[2, 3].simplify())

        fx = sp.lambdify((l1, l2, theta1, theta2, theta3, theta4, pi), px, 'numpy')
        fy = sp.lambdify((l1, l2, theta1, theta2, theta3, theta4, pi), py, 'numpy')
        fz = sp.lambdify((l1, l2, theta1, theta2, theta3, theta4, pi), pz, 'numpy')

        m03 = (m01 * m12 * m23)
        px = sp.trigsimp(m03[0, 3].simplify())
        py = sp.trigsimp(m03[1, 3].simplify())
        pz = sp.trigsimp(m03[2, 3].simplify())

        fxx = sp.lambdify((l1, l2, theta1, theta2, theta3, pi), px, 'numpy')
        fyy = sp.lambdify((l1, l2, theta1, theta2, theta3, pi), py, 'numpy')
        fzz = sp.lambdify((l1, l2, theta1, theta2, theta3, pi), pz, 'numpy')

        return fx, fy, fz, fxx, fyy, fzz

    def inertia(self):
        x_upper = (self.position[0, :] + self.position[1, :]) / 2
        x_lower = (self.position[1, :] + self.position[2, :]) / 2
        Ilower = np.linalg.norm(x_lower - self.position[1, :]) * self.mass[0]
        Iupper = np.linalg.norm(x_upper) * self.mass[0] + np.linalg.norm(x_lower) * self.mass[1]
        return np.asarray([Iupper, Ilower])

    def intertia_to_angular_acceleration(self, torque):
        i = self.inertia()
        I = np.asarray([i[0], i[0], i[0], i[1]])
        acc = torque / I
        return acc

    def step(self, action):
        acc = self.intertia_to_angular_acceleration(action)
        self.theta += self.angular_velocity * self.dt + 0.5 * acc * self.dt ** 2
        self.angular_velocity += acc * self.dt
        self.position[1, 0] = self.x_wrist(self.length[0], self.length[1], self.theta[0], self.theta[1], self.theta[2],
                                           self.theta[3], np.pi)
        self.position[1, 1] = self.y_wrist(self.length[0], self.length[1], self.theta[0], self.theta[1], self.theta[2],
                                           self.theta[3], np.pi)
        self.position[1, 2] = self.z_wrist(self.length[0], self.length[1], self.theta[0], self.theta[1], self.theta[2],
                                           self.theta[3], np.pi)
        self.position[2, 0] = self.x_elbow(self.length[0], self.length[1], self.theta[0], self.theta[1], self.theta[2],
                                           np.pi)
        self.position[2, 1] = self.y_elbow(self.length[0], self.length[1], self.theta[0], self.theta[1], self.theta[2],
                                           np.pi)
        self.position[2, 2] = self.z_elbow(self.length[0], self.length[1], self.theta[0], self.theta[1], self.theta[2],
                                           np.pi)


if __name__ == '__main__':
    action_space = spaces.Box(0, 1, shape=(4,))
    arm = Arm()
    n_frames = 120
    shoulder = np.zeros((n_frames, 1, 3))
    wrist = np.zeros((n_frames, 1, 3))
    elbow = np.zeros((n_frames, 1, 3))

    for i in range(n_frames):
        action = action_space.sample()
        arm.step(action)
        shoulder[i, 0, :] = arm.position[0, :]
        wrist[i, 0, :] = arm.position[1, :]
        elbow[i, 0, :] = arm.position[2, :]

    v = ArmViewer()
    v.setup_run(shoulder, elbow, wrist)
    v.run()
