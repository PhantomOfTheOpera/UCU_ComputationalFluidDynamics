import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm

times = 200
n = "55"
N = "50000"
path = "explosion.af"
command = "./src/arrayfire/build/main"


results = []
for i in tqdm(range(times)):
    results.append(float(str(subprocess.run([command, n, N, path, "CUDA"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout)[2:-3].strip().split("\\n")[-1]))

with open("results_af_10000.txt", "w") as f:
    f.writelines(map(str, results))

print(results)
plt.hist(results, bins=20)
plt.savefig("test.png")