import os

hipified_sources = [
    "cupy_backends/cuda/libs/curand_hip.pyx",
    "cupy_backends/cuda/libs/nvrtc_hip.pyx",
]

def modify_hip_gen_code(path):
    fs = open(path, 'r')
    lines = fs.readlines()
    fs.close()

    for j in range(len(lines)):
        line = lines[j]
        if "CUPY_USE_GEN_HIP_CODE == 0" in line:
            break
        if "CUPY_USE_GEN_HIP_CODE" in line:
            line = line.replace("CUPY_USE_GEN_HIP_CODE", "CUPY_USE_GEN_HIP_CODE == 0")
            lines[j] = line
            break

    fs_write = open(path, 'w')
    fs_write.writelines(lines)
    fs_write.close()
    

def post_process_hipified_files(source_root):
    for source in hipified_sources:
        path = os.path.join(source_root, source)
        modify_hip_gen_code(path)
    print ("OK: modified all the hipified files")
