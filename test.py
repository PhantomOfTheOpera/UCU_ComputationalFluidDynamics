import torch
import numpy as np


gamma = torch.Tensor([7/5]).cuda()
k = torch.Tensor([200]).cuda()
R = torch.Tensor([8.31]).cuda()
mu = torch.Tensor([0.029]).cuda()
c = torch.Tensor([R / ((gamma - 1) * mu)]).cuda()
v_sound = torch.Tensor([343]).cuda()


def U2params(U):
    ro = U[0]
    vx = U[1] / ro
    vx = U[2] / ro
    e = (U[3] - (U[1] ** 2 + U[2] ** 2) / (2 * ro)) / ro

    p = (gamma - 1) * e
    T = e / (ro * c)

    return ro, vx, vx, e, p, T


def params2U(ro, vx, vy, T):
    U = torch.zeros((4)).cuda()
    U[0] = ro
    U[1] = vx * ro
    U[2] = vy * ro
    U[3] = (ro) * ((T) * (ro) * c)

    return U

def params2U_parallel(ro, vx, vy, T):
    U = torch.zeros((n, n, 4)).cuda()
    U[:,:,0] = ro
    U[:,:,1] = vx * ro
    U[:,:,2] = vy * ro
    U[:,:,3] = ro * T * ro * c

    return U

n = 503
n += 2
t = 20000

U = params2U_parallel(1.25, 0, 0, 300)

dx = torch.Tensor([0.001]).cuda()
dy = torch.Tensor([0.001]).cuda()


q = torch.zeros((n, n)).cuda()
fx = torch.zeros((n, n)).cuda()
fy = torch.zeros((n, n)).cuda()

U[2 * ((1* n // 4) // 2), 2 * ((5 * n // 8) // 2)] = params2U(1.25, 0, 0, 400)
U[2 * ((3 * n // 4) // 2), 2 * ((2 * n // 9) // 2)] = params2U(1.25, 0, 0, 470)
U[2 * ((7 * n // 11) // 2), 2 * ((4 * n // 9) // 2)] = params2U(1.25, 0, 0, 650)
U[2 * ((11 * n // 13) // 2), 2 * ((5 * n // 7) // 2)] = params2U(1.25, 0, 0, 90)


def updateU(U):

    dt = torch.Tensor([0.02 * (0
                            +abs(U[:,:,1] / U[:,:,0]).max()/dx
                            +abs(U[:,:,2] / U[:,:,0]).max()/dy
                            + v_sound * (1/dx**2)**(0.5)
                        )**(-1)]).cuda()
    Ucor = U.clone()

    Ucor[2:n-1:2, 2:n-1:2, 0] = (U[2:n-1:2, 2:n-1:2, 0] 
                    - dt * (U[2:n-1:2, 2:n-1:2, 0] + U[4:n+1:2, 2:n-1:2, 0]) * U[3:n:2, 2:n-1:2, 0] / (dx * 2)
                    + dt * (U[2:n-1:2, 2:n-1:2, 0] + U[0:n-3:2, 2:n-1:2, 0]) * U[1:n-2:2, 2:n-1:2, 0] / (dx * 2)
                    - dt * (U[2:n-1:2, 2:n-1:2, 0] + U[2:n-1:2, 4:n+1:2, 0]) * U[2:n-1:2, 3:n:2, 0] / (dy * 2)
                    + dt * (U[2:n-1:2, 2:n-1:2, 0] + U[2:n-1:2, 0:n-3:2, 0]) * U[2:n-1:2, 1:n-2:2, 0] / (dy * 2)
            )

    Ucor[2:n-1:2, 2:n-1:2, 3] = (U[2:n-1:2, 2:n-1:2, 3]
                    - dt * gamma * (U[4:n+1:2, 2:n-1:2, 3] + U[2:n-1:2, 2:n-1:2, 3]) * U[3:n:2, 2:n-1:2, 1] / (dx * 2)
                    + dt * gamma * (U[0:n-3:2, 2:n-1:2, 3] + U[2:n-1:2, 2:n-1:2, 3]) * U[1:n-2:2, 2:n-1:2, 1] / (dx * 2)
                    - dt * gamma * (U[2:n-1:2, 4:n+1:2, 3] + U[2:n-1:2, 2:n-1:2, 3]) * U[2:n-1:2, 3:n:2, 2] / (dy * 2)
                    + dt * gamma * (U[2:n-1:2, 0:n-3:2, 3] + U[2:n-1:2, 2:n-1:2, 3]) * U[2:n-1:2, 1:n-2:2, 2] / (dy * 2)
                    + dt * U[2:n-1:2, 2:n-1:2, 0] * q[2:n-1:2, 2:n-1:2]
                    # + dt * U[2:n-1:2, 2:n-1:2, 0] * (fx[2:n-1:2, 2:n-1:2] * (U[3:n:2, 2:n-1:2, 1] + U[1:n-2:2, 2:n-1:2, 1]) / 2 + fy[2:n-1:2, 2:n-1:2] * (U[2:n-1:2, 3:n:2, 1] + U[2:n-1:2, 1:n-2:2, 1]) / 2)
            )

    Ucor[2:n-1:2, 2:n-1:2, 3] = (Ucor[2:n-1:2, 2:n-1:2, 3]
                    + dt * k * (U[4:n+1:2, 2:n-1:2, 3] / (U[4:n+1:2, 2:n-1:2, 0] * c) - U[2:n-1:2, 2:n-1:2, 3] / (U[2:n-1:2, 2:n-1:2, 0] * c)) / dx**2
                    + dt * k * (U[0:n-3:2, 2:n-1:2, 3] / (U[0:n-3:2, 2:n-1:2, 0] * c) - U[2:n-1:2, 2:n-1:2, 3] / (U[2:n-1:2, 2:n-1:2, 0] * c)) / dx**2
                    + dt * k * (U[2:n-1:2, 4:n+1:2, 3] / (U[2:n-1:2, 4:n+1:2, 0] * c) - U[2:n-1:2, 2:n-1:2, 3] / (U[2:n-1:2, 2:n-1:2, 0] * c)) / dy**2
                    + dt * k * (U[2:n-1:2, 0:n-3:2, 3] / (U[2:n-1:2, 0:n-3:2, 0] * c) - U[2:n-1:2, 2:n-1:2, 3] / (U[2:n-1:2, 2:n-1:2, 0] * c)) / dy**2
            )


    p3 = U[2:n:2, 2:n-1:2, 3] * (gamma - 1)
    p4 = U[0:n-2:2, 2:n-1:2, 3] * (gamma - 1)
    Ucor[1:n-1:2, 2:n-1:2, 1] = (U[1:n-1:2, 2:n-1:2, 1]
        - 2 * dt * p3 / (dx * (U[2:n:2, 2:n-1:2, 0] + U[0:n-2:2, 2:n-1:2, 0]) / 2)
        + 2 * dt * p4 / (dx * (U[2:n:2, 2:n-1:2, 0] + U[0:n-2:2, 2:n-1:2, 0]) / 2)
        + dt * fx[1:n-1:2, 2:n-1:2]
    ) 
    Ucor[1,:,1] = 0
    Ucor[n-2,:,1] = 0
    Ucor[:,1,1] = 0
    Ucor[:,n-2:1] = 0

    p5 = U[2:n-1:2, 2:n:2, 3] * (gamma - 1)
    p6 = U[2:n-1:2, 0:n-2:2, 3] * (gamma - 1)
    Ucor[2:n-1:2, 1:n-1:2, 2] = (U[2:n-1:2, 1:n-1:2, 2]
        - 2 * dt * p5 / (dy * (U[2:n-1:2, 2:n:2, 0] + U[2:n-1:2, 0:n-2:2, 0]) / 2)
        + 2 * dt * p6 / (dy * (U[2:n-1:2, 2:n:2, 0] + U[2:n-1:2, 0:n-2:2, 0]) / 2)
        + dt * fy[2:n-1:2, 1:n-1:2]
    )
    Ucor[1,:,2] = 0
    Ucor[n-2,:,2] = 0
    Ucor[:,1,2] = 0
    Ucor[:,n-2:2] = 0

    return Ucor, dt

surfs = []
for i in range(t):

    U, dt = updateU(U)
    print(i)

    if i % 50 == 0:
        surfs.append(U[::2, ::2, 3] - 335802)


lines = []
for idx, surf in enumerate(surfs):
    print(idx / len(surfs))
    l = len(surf)
    surf_str = np.char.add(surf.cpu().numpy().astype(str), "\n")
    surf_str[:,l-1] = np.char.add(surf_str[:,l-1], "\n")
    surf_x = np.char.add(np.array([list(range(l)) for i in range(l)]).astype(str), " ")
    surf_y = surf_x.T.astype(str)

    surf_new = np.char.add(np.char.add(surf_x, surf_y), surf_str).flatten()
    lines.extend(list(surf_new))
    lines.append("\n\n")

with open("data.txt", "w") as f:
    f.writelines(lines)
