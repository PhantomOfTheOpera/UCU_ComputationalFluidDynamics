#!/home/fedynyak/miniconda3/envs/ntire/bin/python
import time
import torch
import numpy as np
import tqdm
import pandas as pd
import os
import random
import sys


device = "cuda"


class Fluid3D:

    gamma = torch.Tensor([7/5]).to(device)
    k = torch.Tensor([300]).to(device)
    R = torch.Tensor([8.31]).to(device)
    mu = torch.Tensor([0.029]).to(device)
    c = torch.Tensor([R / ((gamma - 1) * mu)]).to(device)
    v_sound = torch.Tensor([343]).to(device)

    def __init__(self, n):

        self.dx = torch.Tensor([0.001]).to(device)
        self.dy = torch.Tensor([0.001]).to(device)
        self.dz = torch.Tensor([0.001]).to(device)

        self.q = torch.zeros((n, n, n)).to(device)
        self.fx = torch.zeros((n, n, n)).to(device)
        self.fy = torch.zeros((n, n, n)).to(device)
        self.fz = torch.zeros((n, n, n)).to(device)

        self.n = n
        self.U = self.params2U_parallel(self.n, 1.25, 0, 0, 0, 300)

        self.surfs4d = []
        self.vecs4d = []

        self.U[n//2 + 1, n//2 + 1, n//2 + 1, :] = self.params2U_parallel(1, 1.25, 0, 0, 0, 400)

    def params2U_parallel(self, n, ro, vx, vy, vz, T):

        U = torch.zeros((n, n, n, 5)).to(device)
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

        dt = torch.Tensor([0.02 * (0
                            + abs(U[:,:,:,1]).max()/self.dx
                            + abs(U[:,:,:,2]).max()/self.dy
                            + abs(U[:,:,:,3]).max()/self.dz
                            + (self.v_sound * (1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2)**0.5))**-1]).to(device)
        
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


if __name__ == "__main__":

    nargs = len(sys.argv)

    N = int(sys.argv[2])
    n = int(sys.argv[1])
    logfreq = int(sys.argv[3])
    savefreq = int(sys.argv[4])
    outpath = "data/torchsim/exp_1"
    disable_tqdm = True

    os.makedirs(outpath, exist_ok=True)

    fluid = Fluid3D(n)
    start = time.time()
    for idx in tqdm.tqdm(range(N), disable=True):

        corrector = fluid._update(fluid.U)
        predictor = fluid._update(corrector)

        fluid.U = 0.5 * (corrector + predictor)

        
