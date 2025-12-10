# build mlir-air and allo

```bash
conda create -n air python=3.12
conda activate air
# Install essential python packages
python3 -m pip install --upgrade pip
python3 -m pip install -r utils/requirements.txt
# Install python packages needed by MLIR-AIE's python bindings
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r utils/requirements_extras.txt
```

if working on xdna2, maybe update `install/python/air/backend/xrt.py` L141 to `target_device = "npu2"`
```bash
# ./utils/build-mlir-air-using-wheels.sh <xrt_dir> [build_dir] [install_dir]
./utils/build-mlir-air-using-wheels.sh /opt/xilinx/xrt
```

export as follows:
```bash
export MLIR_AIR_INSTALL_DIR=/home/sf668/workspace/mlir-air/install
export MLIR_AIE_INSTALL_DIR=/home/sf668/miniconda3/envs/air/lib/python3.12/site-packages/mlir_aie
export LLVM_INSTALL_DIR=/home/sf668/workspace/mlir-air/my_install/mlir

export PATH=${MLIR_AIR_INSTALL_DIR}/bin:${MLIR_AIE_INSTALL_DIR}/bin:${LLVM_INSTALL_DIR}/bin:${PATH} 
export PYTHONPATH=${MLIR_AIR_INSTALL_DIR}/python:${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH} 
export LD_LIBRARY_PATH=${MLIR_AIR_INSTALL_DIR}/lib:${MLIR_AIE_INSTALL_DIR}/lib:${LLVM_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
```

Then build allo (same environment, use different llvm-project)

my current working setup script:
```txt
#!/bin/bash
export XILINX_LOC=/opt/xilinx/Vitis/2024.2
export AIETOOLS_ROOT=$XILINX_LOC/aietools
export PATH=$PATH:${AIETOOLS_ROOT}/bin:$XILINX_LOC/bin
export VITIS=${XILINX_LOC}
export XILINX_VITIS=${XILINX_LOC}
export VITIS_ROOT=${XILINX_LOC}
source ${XILINX_LOC}/settings64.sh
echo "gcc is now $(which gcc)"
echo "make is now $(which make)"
echo "cmake is now $(which cmake)"
source /opt/xilinx/xrt/setup.sh
echo "xrt setup finished. Available devices:"

export NPU2=1
export PATH=/home/sf668/workspace/allo/externals/llvm-project/build/bin:$PATH
export LLVM_BUILD_DIR=/home/sf668/workspace/allo/externals/llvm-project/build

export PATH=/home/sf668/miniconda3/envs/air/lib/python3.12/site-packages/mlir_aie/bin:$PATH
export MLIR_AIE_INSTALL_DIR=/home/sf668/miniconda3/envs/air/lib/python3.12/site-packages/mlir_aie
export PEANO_INSTALL_DIR=/home/sf668/miniconda3/envs/air/lib/python3.12/site-packages/llvm-aie
export MLIR_AIE_EXTERNAL_KERNEL_DIR=/home/sf668/usr//mlir-aie/aie_kernels/
export RUNTIME_LIB_DIR=/home/sf668/usr//mlir-aie/runtime_lib/
export PYTHONPATH=/home/sf668/miniconda3/envs/air/lib/python3.12/site-packages/mlir_aie/python:$PYTHONPATH
export MLIR_AIR_INSTALL_DIR=/home/sf668/workspace/mlir-air/install
export LLVM_INSTALL_DIR=/home/sf668/workspace/mlir-air/my_install/mlir
export PATH=${MLIR_AIR_INSTALL_DIR}/bin:${MLIR_AIE_INSTALL_DIR}/bin:${LLVM_INSTALL_DIR}/bin:${PATH}
export PYTHONPATH=${MLIR_AIR_INSTALL_DIR}/python:${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${MLIR_AIR_INSTALL_DIR}/lib:${MLIR_AIE_INSTALL_DIR}/lib:${LLVM_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
```