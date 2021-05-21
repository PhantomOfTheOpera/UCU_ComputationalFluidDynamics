from cgitb import enable
import subprocess
import matplotlib.pyplot as plt
import tqdm


x = []
y = []
executable = ["python3", "torch_3D.py"]
with open("bench_pytorch_gpu.txt", "w") as f: 
    for n in tqdm.tqdm(range(7, 155, 16)):
        out = subprocess.run([*executable, str(n), str(25), str(25), str(10**6)], capture_output=True)
        # print(n, float(str(out.stdout.strip().split()[1])[2:-1]), file=f)
        print(out)


# f = plt.figure(figsize=(10, 10))
# with open("bench_af.txt") as f:
#     x, y = zip(*list(map(lambda x: [float(x.strip().split()[0]), float(x.strip().split()[1])**(-1/3)], f.readlines())))
# plt.plot(x, y, linewidth=1)
# # plt.yscale("log")
# plt.savefig("graph.png", dpi=200)
