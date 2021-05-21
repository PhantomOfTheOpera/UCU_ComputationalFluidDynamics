import matplotlib.pyplot as plt


f = plt.figure(figsize=(10, 10))
with open("stats/results_gpu.txt") as f:
    y = list(map(lambda x: float(x.strip())**(-1/3), f.readlines()))
s = list(range(13, 150, 2)) + list(range(151, 500, 20))
s = s[:81]
plt.plot(s, y, linewidth=1)
plt.savefig("graph_torch.png", dpi=200)
