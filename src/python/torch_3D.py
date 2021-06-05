#!/home/fedynyak/miniconda3/envs/ntire/bin/python
import time
import torch
import sys


class Fluid3D:

    def __init__(self, n, device, init_state):

        self.device = device

        self.gamma = torch.Tensor([7/5]).to(device)
        self.k = torch.Tensor([300]).to(device)
        self.R = torch.Tensor([8.31]).to(device)
        self.mu = torch.Tensor([0.029]).to(device)
        self.c = torch.Tensor([self.R / ((self.gamma - 1) * self.mu)]).to(device)
        self.v_sound = torch.Tensor([343]).to(device)

        self.dx = torch.Tensor([0.001]).to(device)
        self.dy = torch.Tensor([0.001]).to(device)
        self.dz = torch.Tensor([0.001]).to(device)

        self.q = torch.zeros((n, n, n)).to(device)
        self.fx = torch.zeros((n, n, n)).to(device)
        self.fy = torch.zeros((n, n, n)).to(device)
        self.fz = torch.zeros((n, n, n)).to(device)

        self.n = n
        self.U = torch.zeros((n, n, n, 5))
        self.U_predictor = torch.zeros((n, n, n, 5))
        self.U_corrector = torch.zeros((n, n, n, 5))

        self.read_init_state(init_state)

        self.U = self.U.to(device)
        self.U_predictor = self.U_predictor.to(device)
        self.U_corrector = self.U_corrector.to(device)


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
                    # + dt * U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] * self.q[2:n-1:2, 2:n-1:2, 2:n-1:2]
                    # + dt * U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] * (self.fx[2:n-1:2, 2:n-1:2, 2:n-1:2] * (U[3:n:2, 2:n-1:2, 2:n-1:2, 1] + U[1:n-2:2, 2:n-1:2, 2:n-1:2, 1]) / 2 
                    #                                         + self.fy[2:n-1:2, 2:n-1:2, 2:n-1:2] * (U[2:n-1:2, 3:n:2, 2:n-1:2, 2] + U[2:n-1:2, 1:n-2:2, 2:n-1:2, 2]) / 2 
                    #                                         + self.fz[2:n-1:2, 2:n-1:2, 2:n-1:2] * (U[2:n-1:2, 2:n-1:2, 3:n:2, 3] + U[2:n-1:2, 2:n-1:2, 1:n-2:2, 3]) / 2)
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

    def _update(self, U, Unew):

        n = self.n

        dt = torch.Tensor([0.02 * (0
                            + abs(U[:,:,:,1]).max()/self.dx
                            + abs(U[:,:,:,2]).max()/self.dy
                            + abs(U[:,:,:,3]).max()/self.dz
                            + (self.v_sound * (1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2)**0.5))**-1]).to(self.device)
        
        self.update_RO(Unew, U, dt)
        self.update_E(Unew, U, dt)
        self.update_V(Unew, U, dt)

        Unew[1,:,:,1] = 0
        Unew[n-2,:,:,1] = 0

        Unew[:,1,:,2] = 0
        Unew[:,n-2,:2] = 0

        Unew[:,:,1,3] = 0
        Unew[:,:,n-2,3] = 0
    
    def step(self):
        self._update(self.U, self.U_predictor)
        self._update(self.U_predictor, self.U_corrector)
        self.U = (self.U + self.U_corrector) * 0.5
    
    def read_init_state(self, path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                x, y, z, ro, vx, vy, vz, e = map(float, line.split())
                x, y, z = map(int, [x, y, z])
                self.U[x, y, z] = torch.Tensor([ro, vx, vy, vz, e])
                self.U_predictor[x, y, z] = torch.Tensor([ro, vx, vy, vz, e])
                self.U_corrector[x, y, z] = torch.Tensor([ro, vx, vy, vz, e])


if __name__ == "__main__":

    nargs = len(sys.argv)

    n = int(sys.argv[1])
    N = int(sys.argv[2])
    init_state = sys.argv[3]
    backend = sys.argv[4]

    if backend == "CPU":
        device = "cpu"
    elif backend == "CUDA":
        device = "cuda"
    else:
        print("BACKEND NOT SUPPORTED")
        exit(1)

    fluid = Fluid3D(n, device, init_state)
    start = time.time()
    for idx in range(N):
        fluid.step()
    end = time.time()
    print(end - start)