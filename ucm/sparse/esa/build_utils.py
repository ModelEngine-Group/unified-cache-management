import os
import sysconfig
import subprocess
import torch

def build_shared(src_files, target, mode = "release"):
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    cuda_inc = os.path.join(cuda_home, "include")
    cuda_lib = os.path.join(cuda_home, "lib64")
    if not os.path.isdir(cuda_inc) or not os.path.isdir(cuda_lib):
        raise SystemExit(f"CUDA not found. Set CUDA_HOME or install to {cuda_home}")

    py_inc = sysconfig.get_paths()["include"]

    # Torch include/library paths
    t_inc = include_paths()  # e.g., [.../torch/include, .../torch/include/torch/csrc/api/include]
    t_lib = library_paths()  # e.g., [.../torch/lib]

    # ABI flag must match the one PyTorch was built with
    cxx11_abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 1)
    abi_macro = f"-D_GLIBCXX_USE_CXX11_ABI={int(cxx11_abi)}"
    target_name = target.split(".")[0]
    print("target_name: ", target_name)

    print("==== nvcc_compile")
    cmd = [
        "nvcc",
        "-std=c++17",
        "-Xcompiler",
        "-fPIC",
        "-shared",
        "-I" + py_inc,
        "-I" + cuda_inc,
        *[f"-I{p}" for p in t_inc],
        # ABI macro
        abi_macro,
        f"-DTORCH_EXTENSION_NAME={target_name}",
        # libs
        "-L" + cuda_lib,
        *[f"-L{p}" for p in t_lib],
        "-Xlinker", "-rpath", "-Xlinker", cuda_lib,
        *[arg for p in t_lib for arg in ("-Xlinker", "-rpath", "-Xlinker", p)],
        # link against torch and CUDA runtime
        "-lc10",
        "-lc10_cuda",
        "-ltorch_cpu",
        "-ltorch_cuda",
        "-ltorch",
        "-ltorch_python",
        "-lcudart",
    ]
    if mode == "release":
        cmd.append("-O3")
    else:
        cmd.extend(["-g", "-G", "-lineinfo", "-DTORCH_USE_CUDA_DSA"])

    assert isinstance(src_files, list) or isinstance(src_files, tuple)
    cmd.extend(src_files)
    cmd.extend(["-o", target])
    print("Building so with:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # build_shared(["./diff_map_thrust_pybind.cu"], "diff_map.so")
    build_shared(["./esa_interface.cc", "./esa_kernels.cu", "./esa_sm_copy.cu"], "esa_interface.so")
