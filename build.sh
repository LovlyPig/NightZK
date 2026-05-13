#!/bin/bash
set -e

ANA_PATH="/public/software/apps/anaconda3/2023.09"
DTK_PATH="/public/software/sghpc_sdk/Linux_x86_64/26.3/dtk/dtk-25.04.4"

if command -v module > /dev/null 2>&1; then
    # 仅在未加载时执行加载
    module list 2>&1 | grep -q "anaconda3" || module load anaconda3/2023.09
    module list 2>&1 | grep -q "sghpc-mpi-gcc" || module load sghpc-mpi-gcc/26.3
else
    echo "Warning: 'module' command not found. Assuming environment is pre-configured."
fi

export HIP_COMPILER=clang
export ROCM_PATH=${ROCM_PATH:-$DTK_PATH}
export HIP_PATH=$ROCM_PATH
export CMAKE_PREFIX_PATH=$ROCM_PATH/dcc/comgr/lib64/cmake:$ANA_PATH:$CMAKE_PREFIX_PATH

rm -rf build && mkdir build && cd build

cmake .. \
	-DWITH_PROCPS=OFF

make -j$(nproc)
	
	
