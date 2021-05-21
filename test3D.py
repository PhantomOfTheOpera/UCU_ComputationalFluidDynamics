import torch
import numpy as np


class Fluid3D:

    gamma = torch.Tensor([7/5]).cuda()
    k = torch.Tensor([150]).cuda()
    R = torch.Tensor([8.31]).cuda()
    mu = torch.Tensor([0.029]).cuda()
    c = torch.Tensor([R / ((gamma - 1) * mu)]).cuda()
    v_sound = torch.Tensor([343]).cuda()

    def __init__(self, n):

        self.dx = torch.Tensor([0.001]).cuda()
        self.dy = torch.Tensor([0.001]).cuda()
        self.dz = torch.Tensor([0.001]).cuda()

        self.q = torch.zeros((n, n, n)).cuda()
        self.fx = torch.zeros((n, n, n)).cuda()
        self.fy = torch.zeros((n, n, n)).cuda()
        self.fz = torch.zeros((n, n, n)).cuda()

        self.n = n
        self.U = self.params2U_parallel(self.n, 1.25, 0, 0, 0, 300)

        self.surfs4d = []
        self.vecs4d = []

        self.fz -= 10**4
        # self.U[n//2 + 1, n//2 + 1, n//2 + 1, :] = self.params2U_parallel(1, 1.25, 0, 0, 0, 400)
        self.q[n//2 + 1, n//2 + 1, n//2 + 1] = 10**10

    def params2U_parallel(self, n, ro, vx, vy, vz, T):

        U = torch.zeros((n, n, n, 5)).cuda()
        U[:,:,:,0] = ro
        U[:,:,:,1] = vx 
        U[:,:,:,2] = vy
        U[:,:,:,3] = vz
        U[:,:,:,4] = ro * T * ro * self.c

        return U

    def update_RO(self, Unew, U,  dt):

        n = self.n

        Unew[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] = (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] 
                    - dt * (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] + U[4:n+1:2, 2:n-1:2, 2:n-1:2, 0]) * U[3:n:2, 2:n-1:2, 2:n-1:2, 1] / (self.dx * 2)
                    + dt * (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] + U[0:n-3:2, 2:n-1:2, 2:n-1:2, 0]) * U[1:n-2:2, 2:n-1:2, 2:n-1:2, 1] / (self.dx * 2)
                    - dt * (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] + U[2:n-1:2, 4:n+1:2, 2:n-1:2, 0]) * U[2:n-1:2, 3:n:2, 2:n-1:2, 2] / (self.dy * 2)
                    + dt * (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] + U[2:n-1:2, 0:n-3:2, 2:n-1:2, 0]) * U[2:n-1:2, 1:n-2:2, 2:n-1:2, 2] / (self.dy * 2)
                    - dt * (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] + U[2:n-1:2, 2:n-1:2, 4:n+1:2, 0]) * U[2:n-1:2, 2:n-1:2, 3:n:2, 3] / (self.dz * 2)
                    + dt * (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] + U[2:n-1:2, 2:n-1:2, 0:n-3:2, 0]) * U[2:n-1:2, 2:n-1:2, 1:n-2:2, 3] / (self.dz * 2)
            )

    def update_E(self, Unew, U, dt):

        n = self.n

        Unew[2:n-1:2, 2:n-1:2, 2:n-1:2, 4] = (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4]
                    - dt * self.gamma * (U[4:n+1:2, 2:n-1:2, 2:n-1:2, 4] + U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4]) * U[3:n:2, 2:n-1:2, 2:n-1:2, 1] / (self.dx * 2)
                    + dt * self.gamma * (U[0:n-3:2, 2:n-1:2, 2:n-1:2, 4] + U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4]) * U[1:n-2:2, 2:n-1:2, 2:n-1:2, 1] / (self.dx * 2)
                    - dt * self.gamma * (U[2:n-1:2, 4:n+1:2, 2:n-1:2, 4] + U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4]) * U[2:n-1:2, 3:n:2, 2:n-1:2, 2] / (self.dy * 2)
                    + dt * self.gamma * (U[2:n-1:2, 0:n-3:2, 2:n-1:2, 4] + U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4]) * U[2:n-1:2, 1:n-2:2, 2:n-1:2, 2] / (self.dy * 2)
                    - dt * self.gamma * (U[2:n-1:2, 2:n-1:2, 4:n+1:2, 4] + U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4]) * U[2:n-1:2, 2:n-1:2, 3:n:2, 3] / (self.dz * 2)
                    + dt * self.gamma * (U[2:n-1:2, 2:n-1:2, 0:n-3:2, 4] + U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4]) * U[2:n-1:2, 2:n-1:2, 1:n-2:2, 3] / (self.dz * 2)
                    + dt * U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] * self.q[2:n-1:2, 2:n-1:2, 2:n-1:2]
                    + dt * U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] * (self.fx[2:n-1:2, 2:n-1:2, 2:n-1:2] * (U[3:n:2, 2:n-1:2, 2:n-1:2, 1] + U[1:n-2:2, 2:n-1:2, 2:n-1:2, 1]) / 2 
                                                            + self.fy[2:n-1:2, 2:n-1:2, 2:n-1:2] * (U[2:n-1:2, 3:n:2, 2:n-1:2, 2] + U[2:n-1:2, 1:n-2:2, 2:n-1:2, 2]) / 2 
                                                            + self.fz[2:n-1:2, 2:n-1:2, 2:n-1:2] * (U[2:n-1:2, 2:n-1:2, 3:n:2, 3] + U[2:n-1:2, 2:n-1:2, 1:n-2:2, 3]) / 2)
            )

        Unew[2:n-1:2, 2:n-1:2, 2:n-1:2, 4] = (Unew[2:n-1:2, 2:n-1:2, 2:n-1:2, 4]
                        + dt * self.k * (U[4:n+1:2, 2:n-1:2, 2:n-1:2, 4] / (U[4:n+1:2, 2:n-1:2, 2:n-1:2, 0] * self.c) - U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4] / (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] * self.c)) / self.dx**2
                        + dt * self.k * (U[0:n-3:2, 2:n-1:2, 2:n-1:2, 4] / (U[0:n-3:2, 2:n-1:2, 2:n-1:2, 0] * self.c) - U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4] / (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] * self.c)) / self.dx**2
                        + dt * self.k * (U[2:n-1:2, 4:n+1:2, 2:n-1:2, 4] / (U[2:n-1:2, 4:n+1:2, 2:n-1:2, 0] * self.c) - U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4] / (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] * self.c)) / self.dy**2
                        + dt * self.k * (U[2:n-1:2, 0:n-3:2, 2:n-1:2, 4] / (U[2:n-1:2, 0:n-3:2, 2:n-1:2, 0] * self.c) - U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4] / (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] * self.c)) / self.dy**2
                        + dt * self.k * (U[2:n-1:2, 2:n-1:2, 4:n+1:2, 4] / (U[2:n-1:2, 2:n-1:2, 4:n+1:2, 0] * self.c) - U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4] / (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] * self.c)) / self.dz**2
                        + dt * self.k * (U[2:n-1:2, 2:n-1:2, 0:n-3:2, 4] / (U[2:n-1:2, 2:n-1:2, 0:n-3:2, 0] * self.c) - U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4] / (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] * self.c)) / self.dz**2
                )

    def update_V(self, Unew, U, dt):

        n = self.n

        p1 = U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4] * (self.gamma - 1)
        p2 = U[0:n-3:2, 2:n-1:2, 2:n-1:2, 4] * (self.gamma - 1)
        Unew[1:n-2:2, 2:n-1:2, 2:n-1:2, 1] = (U[1:n-2:2, 2:n-1:2, 2:n-1:2, 1]
            - 2 * dt * p1 / (self.dx * (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] + U[0:n-3:2, 2:n-1:2, 2:n-1:2, 0]) / 2)
            + 2 * dt * p2 / (self.dx * (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] + U[0:n-3:2, 2:n-1:2, 2:n-1:2, 0]) / 2)
            + dt * (self.fx[2:n-1:2, 2:n-1:2, 2:n-1:2] + self.fx[0:n-3:2, 2:n-1:2, 2:n-1:2]) / 2
        )

        p1 = U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4] * (self.gamma - 1)
        p2 = U[2:n-1:2, 0:n-3:2, 2:n-1:2, 4] * (self.gamma - 1)

        Unew[2:n-1:2, 1:n-2:2, 2:n-1:2, 2] = (U[2:n-1:2, 1:n-2:2, 2:n-1:2, 2]
            - 2 * dt * p1 / (self.dx * (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] + U[2:n-1:2, 0:n-3:2, 2:n-1:2, 0]) / 2)
            + 2 * dt * p2 / (self.dx * (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] + U[2:n-1:2, 0:n-3:2, 2:n-1:2, 0]) / 2)
            + dt * (self.fy[2:n-1:2, 2:n-1:2, 2:n-1:2] + self.fy[2:n-1:2, 0:n-3:2, 2:n-1:2]) / 2
        )

        p1 = U[2:n-1:2, 2:n-1:2, 2:n-1:2, 4] * (self.gamma - 1)
        p2 = U[2:n-1:2, 2:n-1:2, 0:n-3:2, 4] * (self.gamma - 1)
        Unew[2:n-1:2, 2:n-1:2, 1:n-2:2, 3] = (U[2:n-1:2, 2:n-1:2, 1:n-2:2, 3]
            - 2 * dt * p1 / (self.dx * (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] + U[2:n-1:2, 2:n-1:2, 0:n-3:2, 0]) / 2)
            + 2 * dt * p2 / (self.dx * (U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] + U[2:n-1:2, 2:n-1:2, 0:n-3:2, 0]) / 2)
            + dt * (self.fz[2:n-1:2, 2:n-1:2, 2:n-1:2] + self.fz[2:n-1:2, 2:n-1:2, 0:n-3:2]) / 2
        )

    def _update(self, U):

        dt = torch.Tensor([0.05 * (0
                            + abs(U[:,:,:,1]).max()/self.dx
                            + abs(U[:,:,:,2]).max()/self.dy
                            + abs(U[:,:,:,3]).max()/self.dz
                            + (self.v_sound * (1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2)**0.5))**-1]).cuda()
        
        Unew = torch.clone(U)

        self.update_RO(Unew, U, dt)
        self.update_E(Unew, U, dt)
        self.update_V(Unew, U, dt)

        Unew[0, 1:-1, 1:-1] = Unew[1, 1:-1, 1:-1]
        Unew[1:-1, 0, 1:-1] = Unew[1:-1, 1, 1:-1]
        Unew[1:-1, 1:-1, 0] = Unew[1:-1, 1:-1, 1]

        Unew[-1, 1:-1, 1:-1] = Unew[-2, 1:-1, 1:-1]
        Unew[1:-1, -1, 1:-1] = Unew[1:-1, -2, 1:-1]
        Unew[1:-1, 1:-1, -1] = Unew[1:-1, 1:-1, -2]

        Unew[1,:,:,1] = 0
        Unew[n-2,:,:,1] = 0

        Unew[:,1,:,2] = 0
        Unew[:,n-2,:2] = 0

        Unew[:,:,1,3] = 0
        Unew[:,:,n-2,3] = 0

        return Unew

    def run(self, t):

        sc = 2

        for idx in range(t):

            corrector = self._update(self.U)
            predictor = self._update(corrector)

            self.U = 0.5 * (corrector + predictor)

            # print(float(self.U[1:-1,1:-1,1:-1,4].sum().data))
            print(idx)

            if idx % 100 == 0:
                # print(idx)
                self.surfs4d.append(self.U[::2, ::2, ::2, 4] - 335802)
                self.vecs4d.append((self.U[1::2 * sc, 2::2 * sc, 2::2 * sc, 1],
                                    self.U[2::2 * sc, 1::2 * sc, 2::2 * sc, 2],
                                    self.U[2::2 * sc, 2::2 * sc, 1::2 * sc, 3]))


if __name__ == "__main__":

    t = 40000
    n = 103

    fluid = Fluid3D(n)
    fluid.run(t)

    lines = []
    for idx, vec in enumerate(fluid.vecs4d):
        vx, vy, vz = vec

        # vx = vx[:,n//4,:]
        # vy = vy[:,n//4,:]
        # vz = vz[:,n//4,:]

        print(idx / len(fluid.vecs4d))
        l = len(vx)
        vx_str = np.char.add(vx.cpu().numpy().astype(str), " ")
        vy_str = np.char.add(vy.cpu().numpy().astype(str), " ")
        vz_str = np.char.add(vz.cpu().numpy().astype(str), "\n")
        surf_x, surf_y, surf_z = np.meshgrid(list(range(l)), list(range(l)), list(range(l)))
        # surf_x, surf_y, surf_z = np.meshgrid(list(range(l)), 1, list(range(l)))

        surf_x = np.char.add(surf_x.astype(str), " ")
        surf_y = np.char.add(surf_y.astype(str), " ")
        surf_z = np.char.add(surf_z.astype(str), " ")

        vec_field = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(surf_x, surf_y), surf_z), vx_str), vy_str), vz_str).flatten()
        lines.extend(list(vec_field))
        lines.append("\n\n")

    with open("data4d.txt", "w") as f:
        f.writelines(lines)

    # import pyvista as pv

    # final = (fluid.U[1::2, 2::2, 2::2, 1], fluid.U[2::2, 1::2, 2::2, 2], fluid.U[2::2, 2::2, 1::2, 3])

    # vx, vy, vz = map(lambda x: x.cpu().numpy(), final)

    # l = len(vx)

    # vectors = np.zeros((l**3, 3))
    # vectors[:, 0] = vx.flatten()
    # vectors[:, 1] = vy.flatten()
    # vectors[:, 2] = vz.flatten()

    # mesh = pv.UniformGrid((l, l, l), (.1, .1, .1), (0, 0, 0))

    # mesh['vectors'] = vectors

    # stream, src = mesh.streamlines('vectors', return_source=True,
    #                             terminal_speed=0.0, n_points=200,
    #                             source_radius=0.1)

    # cpos = [(1.2, 1.2, 1.2), (-0.0, -0.0, -0.0), (0.0, 0.0, 1.0)]
    # stream.tube(radius=0.0005).plot(cpos=cpos)

    import matplotlib.pyplot as plt

    u, v = fluid.U[1::2, n//2 + 1, 2::2, 1].cpu().numpy(), fluid.U[2::2, n//2 + 1, 1::2, 3].cpu().numpy()
    x, y = np.meshgrid(list(range(len(u))), list(range(len(u))))
    plt.streamplot(x, y, u, v)
    plt.savefig("tst.png")

# lines = []
# for idx, surf in enumerate(surfs4d):
#     print(idx / len(surfs4d))
#     l = len(surf)
#     surf_str = np.char.add(surf.cpu().numpy().astype(str), "\n")
#     surf_str[:,:,l-1] = np.char.add(surf_str[:,:,l-1], "\n")
#     surf_x, surf_y, surf_z = np.meshgrid(list(range(l)), list(range(l)), list(range(l)))
#     surf_x = np.char.add(surf_x.astype(str), " ")
#     surf_y = np.char.add(surf_y.astype(str), " ")
#     surf_z = np.char.add(surf_z.astype(str), " ")

#     vec_field = np.char.add(np.char.add(np.char.add(surf_x, surf_y), surf_z), surf_str).flatten()
#     lines.extend(list(vec_field))
#     lines.append("\n\n")

# with open("data5d.txt", "w") as f:
#     f.writelines(lines)

# from mayavi import mlab
# import numpy as np
# import os
# import glob

# mlab.options.offscreen = True
# path = "tmp"
# data = surfs4d
# fig_myv = mlab.figure(size=(1000,1000), bgcolor=(1,1,1))


# for idx, srf in enumerate(data):
#     print(idx / len(surfs4d))
#     l = len(srf)
#     x, y, z = np.meshgrid(list(range(l)), list(range(l)), list(range(l)))
#     sf = mlab.pipeline.scalar_field(srf.cpu().numpy())

#     mlab.clf()
#     mlab.pipeline.volume(x, y, z, sf)
#     mlab.axes()
#     mlab.savefig(f"tmp/{idx}.png")

# import imageio
# with imageio.get_writer('mlab.gif', mode='I') as writer:
#     for filename in glob.glob(f"{path}/*.png"):
#         image = imageio.imread(filename)
#         writer.append_data(image)


# import pyvista as pv


# opacity = [0, 0, 0, 0.1, 0.3, 0.6, 1]
# plotter = pv.Plotter(off_screen=True)
# plotter.add_volume(pv.wrap(surfs4d[0].cpu().numpy()), cmap="viridis", opacity=opacity)
# plotter.show(auto_close=False)

# plotter.open_movie("wave.mp4")

# for srf in surfs4d:
#     print(666)
#     vol = pv.wrap(srf.cpu().numpy())
#     plotter.add_volume(vol, cmap="viridis", opacity=opacity)
#     plotter.write_frame()

# plotter.close()
