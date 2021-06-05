import subprocess
from tqdm import tqdm
import os
import yaml


def experiment(cfg, fw, bk):

    for backend, conf in bk['BACKENDS'].items():
        print(fw, backend)
        final_dir = os.path.join(cfg['RESULT_DIR'], f"{fw}_{backend}")
        os.makedirs(final_dir, exist_ok=True)
        with open(os.path.join(final_dir, f"{cfg['SCENARIO']}_{cfg['n']}_{cfg['N']}_{conf['T']}.txt"), "w") as f:
            for i in tqdm(range(conf['T'])):
                stdout = subprocess.run([bk['CMD']] + list(map(str, conf['ARGS'])), stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
                res = float(str(stdout)[2:-3])
                f.write(str(res) + "\n")

    # n = n*2 - 1

    # if framework == "AF":
    #     if backend not in ["OPENCL_GPU", "CUDA"]:
    #         print("NOT SUPPORTED BACKEND")
    #         return
    #     cmd = "./src/arrayfire/build/main"
    #     init_file = os.path.join("init_states", f"explosion_{n}.af")
    #     arguments = [str(n), str(N), init_file, backend]
    #     final_dir = os.path.join(results_dir, f"AF_{backend}")
    #     os.makedirs(final_dir, exist_ok=True)
    #     with open(os.path.join(final_dir, f"exp_{n}_{N}_{times}.txt"), "w") as f:
    #         for i in tqdm(range(times)):
    #             stdout = subprocess.run([cmd] + arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    #             res = float(str(stdout)[2:-3])
    #             f.write(str(res) + "\n")

    # if framework == "CPP":
    #     cmd = "./src/cpp/main"
    #     init_file = os.path.join("init_states", f"explosion_{n}.txt")
    #     if backend not in ["CPU"]:
    #         print("NOT SUPPORTED BACKEND")
    #         return
    #     arguments = [str(n), str(N), "8", init_file]
    #     final_dir = os.path.join(results_dir, f"CPP_{backend}")
    #     os.makedirs(final_dir, exist_ok=True)
    #     with open(os.path.join(final_dir, f"exp_{n}_{N}_{times}.txt"), "w") as f:
    #         for i in tqdm(range(times)):
    #             stdout = subprocess.run([cmd] + arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    #             res = float(str(stdout)[2:-3])
    #             f.write(str(res) + "\n")


    # if framework == "PYTORCH":
    #     if backend not in ["CPU", "CUDA"]:
    #         print("NOT SUPPORTED BACKEND")
    #         return
    #     cmd = "./src/python/torch_3D.py"
    #     init_file = os.path.join("init_states", f"explosion_{n}.txt")
    #     arguments = [str(n), str(N), init_file, backend]
    #     final_dir = os.path.join(results_dir, f"PYTORCH_{backend}")
    #     os.makedirs(final_dir, exist_ok=True)
    #     with open(os.path.join(final_dir, f"exp_{n}_{N}_{times}.txt"), "w") as f:
    #         for i in tqdm(range(times)):
    #             stdout = subprocess.run([cmd] + arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    #             res = float(str(stdout)[2:-3])
    #             f.write(str(res) + "\n")


    # if framework == "MPI":
    #     if backend not in ["CPU", "OPENCL_GPU", "CUDA", "OPENCL_GPU_2", "CUDA_2"]:
    #         print("NOT SUPPORTED BACKEND")
    #         return
    #     cmd = ["/usr/bin/mpirun"]
    #     if backend == "CPU":
    #         np = 1
    #     elif backend in ["OPENCL_GPU", "CUDA"]:
    #         np = 1
    #     else:
    #         np = 2
    #     arguments = ["-np", str(np), "src/mpi/build/mpi_app" ,str(n), str(N), backend]
    #     final_dir = os.path.join(results_dir, f"MPI_{backend}")
    #     os.makedirs(final_dir, exist_ok=True)
    #     with open(os.path.join(final_dir, f"exp_{n}_{N}_{times}.txt"), "w") as f:
    #         for i in tqdm(range(times)):
    #             print(cmd + arguments)
    #             stdout = subprocess.run(cmd + arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    #             res = float(str(stdout)[2:-3])
    #             f.write(str(res) + "\n")

if __name__ == "__main__":

    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])
    def nscale(loader, node):
        seq = loader.construct_sequence(node)
        return seq[0] *2 - 1

    yaml.add_constructor('!join', join)
    yaml.add_constructor('!nscale', nscale)

    with open("exp_cfg.yaml") as f:
        cfg = yaml.load(f)

    for fw, bk in cfg["FRAMEWORKS"].items():
        if bk != None:
            try:
                experiment(cfg, fw, bk)
            except:
                continue