# written by Stefan Leutenegger, TU Munich, November 2021
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


# helper class that just does the radial-tangentialDistortion
class RadialTangentialDistortion:
    def __init__(self, k1, k2, p1, p2, k3=0, k4=0, k5=0, k6=0):
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6

    def distort(self, u_undistorted):
        u_undistorted = u_undistorted.reshape(-1, 2)
        u, v = u_undistorted.T
        r2 = u**2 + v**2

        u_residual = 2 * self.p1 * u * v + self.p2 * (r2 + 2 * u**2)
        v_residual = 2 * self.p1 * (r2 + 2 * v**2) + 2 * self.p2 * u * v
        residual = np.stack([u_residual, v_residual], axis=1)

        radial_coeff = (
            (1 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3)
            / (1 + self.k4 * r2 + self.k5 * r2**2 + self.k6 * r2**3)
        ).reshape(-1, 1)
        return radial_coeff * u_undistorted + residual

    def undistort(self, uDistorted):
        num_points = uDistorted.shape[
            0
        ]  # Axis 0 contains the number of points/images. Pass a single element in an array for a single image
        uUnDistorted = np.zeros_like(uDistorted)
        diff = lambda uUndistortedk, uDistorted_k: np.linalg.norm(
            self.distort(uUndistortedk) - uDistorted_k
        )
        # make it work for many points:
        for k in range(num_points):
            uDistortedk = uDistorted[k, :]
            res = minimize(
                diff, uDistortedk, args=(uDistortedk), method="Nelder-Mead", tol=1e-6
            )
            uUnDistorted[k, :] = res.x
        return uUnDistorted


# now the pinhole camera class
class PinholeCamera:
    def __init__(self, width, height, f1, f2, c1, c2, distortion):
        self.width = width
        self.height = height
        self.f1 = f1
        self.f2 = f2
        self.c1 = c1
        self.c2 = c2
        self.distortion = distortion

    def project(self, x):
        x_dash, is_valid = self.p(x)
        x_ddash = self.d(x_dash)
        u = self.k(x_ddash)

        is_valid_index = np.argwhere(is_valid)[:, 0]
        is_in_image_u = np.logical_and(u[:, 0] >= -0.5, u[:, 0] <= self.width - 0.5)
        is_in_image_v = np.logical_and(u[:, 1] >= -0.5, u[:, 1] <= self.height - 0.5)
        is_in_image = np.logical_and(is_in_image_u, is_in_image_v)
        is_valid[is_valid_index] = is_in_image

        if not np.any(is_valid):
            return [], is_valid
        return u[is_in_image, :], is_valid

    def p(self, x):
        is_z_positive = x[:, 2] > 1e-10
        u = x[is_z_positive, 0] / x[is_z_positive, 2]
        v = x[is_z_positive, 1] / x[is_z_positive, 2]
        return np.stack([u, v], axis=1), is_z_positive

    def p_inverse(self, x_dash):
        return np.stack(
            [x_dash[:, 0], x_dash[:, 1], np.ones(np.size(x_dash[:, 0]))], axis=1
        )

    def d(self, x_dash):
        return self.distortion.distort(x_dash)

    def d_inverse(self, x_ddash):
        return self.distortion.undistort(x_ddash)

    def k(self, x_ddash):
        u = self.f1 * x_ddash[:, 0] + self.c1
        v = self.f2 * x_ddash[:, 1] + self.c2
        return np.stack([u, v], axis=1)

    def k_inverse(self, u):
        x_ddash1 = 1.0 / self.f1 * (u[:, 0] - self.c1)
        x_ddash2 = 1.0 / self.f2 * (u[:, 1] - self.c2)
        return np.stack([x_ddash1, x_ddash2], axis=1)

    def backproject(self, u):
        x_ddash = self.k_inverse(u)
        x_dash = self.d_inverse(x_ddash)
        return self.p_inverse(x_dash)


# now test with a projected cube
b = 1.0  # sidelength
z_distance = 2.0  # distance from the camera along the z-axis
N = 20  # number of points on edge

spacing = np.linspace(0, b, N).reshape(N, 1)
e1 = np.array([-b / 2, -b / 2.0, -b / 2.0 + z_distance]).reshape(1, 3) + spacing.dot(
    np.array([1, 0, 0]).reshape(1, 3)
)
e2 = np.array([-b / 2, -b / 2.0, b / 2.0 + z_distance]).reshape(1, 3) + spacing.dot(
    np.array([1, 0, 0]).reshape(1, 3)
)
e3 = np.array([-b / 2, b / 2.0, -b / 2.0 + z_distance]).reshape(1, 3) + spacing.dot(
    np.array([1, 0, 0]).reshape(1, 3)
)
e4 = np.array([-b / 2, b / 2.0, b / 2.0 + z_distance]).reshape(1, 3) + spacing.dot(
    np.array([1, 0, 0]).reshape(1, 3)
)
e5 = np.array([b / 2, -b / 2.0, -b / 2.0 + z_distance]).reshape(1, 3) + spacing.dot(
    np.array([0, 1, 0]).reshape(1, 3)
)
e6 = np.array([b / 2, -b / 2.0, b / 2.0 + z_distance]).reshape(1, 3) + spacing.dot(
    np.array([0, 1, 0]).reshape(1, 3)
)
e7 = np.array([-b / 2, -b / 2.0, -b / 2.0 + z_distance]).reshape(1, 3) + spacing.dot(
    np.array([0, 1, 0]).reshape(1, 3)
)
e8 = np.array([-b / 2, -b / 2.0, b / 2.0 + z_distance]).reshape(1, 3) + spacing.dot(
    np.array([0, 1, 0]).reshape(1, 3)
)
e9 = np.array([b / 2, -b / 2.0, -b / 2.0 + z_distance]).reshape(1, 3) + spacing.dot(
    np.array([0, 0, 1]).reshape(1, 3)
)
e10 = np.array([b / 2, b / 2.0, -b / 2.0 + z_distance]).reshape(1, 3) + spacing.dot(
    np.array([0, 0, 1]).reshape(1, 3)
)
e11 = np.array([-b / 2, -b / 2.0, -b / 2.0 + z_distance]).reshape(1, 3) + spacing.dot(
    np.array([0, 0, 1]).reshape(1, 3)
)
e12 = np.array([-b / 2, b / 2.0, -b / 2.0 + z_distance]).reshape(1, 3) + spacing.dot(
    np.array([0, 0, 1]).reshape(1, 3)
)
edges = np.concatenate([e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12], axis=0)

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(edges[:, 0], edges[:, 1], edges[:, 2])
_ = ax.set_xlabel("x")
_ = ax.set_ylabel("y")
_ = ax.set_zlabel("z")
plt.show(block=True)

# create a plausible pinhole camera model, VGA resolution
pinholeCamera = PinholeCamera(
    640,
    480,
    450,
    450,
    319.5,
    239.5,
    RadialTangentialDistortion(-0.3, 0.1, -0.0001, -0.00005),
)

u, is_valid = pinholeCamera.project(edges)

fig, ax = plt.subplots()
ax.scatter(u[:, 0], u[:, 1])
_ = ax.set_xlim(0, 640)
_ = ax.set_ylim(0, 480)
plt.show(block=True)

# here is a unit test
distortion = RadialTangentialDistortion(-0.3, 0.1, -0.0001, -0.00005)
pinholeCamera = PinholeCamera(640, 480, 450, 450, 319.5, 239.5, distortion)

success = True
for i in range(0, 1000):
    # generate random visible point in image
    u_1 = np.random.uniform(-0.49, 639.49)
    u_2 = np.random.uniform(-0.49, 479.49)
    # back-project and assign random distance
    randomPoint = np.array([[u_1, u_2]])
    ray = pinholeCamera.backproject(np.array([[u_1, u_2]]))
    # project again
    point, is_valid = pinholeCamera.project(ray)
    # check the projection is the same as the generated initial image point
    if np.linalg.norm(point - randomPoint) > 0.001:
        success = False
        break

if success:
    print("[PASSED]")
else:
    print("[FAILED]")

# because we wrote the functions above in a way that they accept arrays of points,
# we can do this unit test in a more elegant / Pythonic way, too:
distortion = RadialTangentialDistortion(-0.3, 0.1, -0.0001, -0.00005)
pinholeCamera = PinholeCamera(640, 480, 450, 450, 319.5, 239.5, distortion)

# generate random visible point in image
u_1 = np.random.uniform(-0.49, 639.49, size=1000)
u_2 = np.random.uniform(-0.49, 479.49, size=1000)
randomPoint = np.stack([u_1, u_2], axis=1)

# back-project and assign random distance
ray = pinholeCamera.backproject(randomPoint)

# project again
point, is_valid = pinholeCamera.project(ray)

# check the projection is the same as the generated initial image point
error = np.linalg.norm(point - randomPoint[is_valid], axis=1)
assert np.all(error < 0.001)
