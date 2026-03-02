#!/bin/bash
# 解决 GLIBCXX_3.4.29 not found：让运行时优先使用 conda 自带的 libstdc++
#
# 原因：rdkit、PIL 等编译时用了较新 GCC，需要 GLIBCXX_3.4.29；系统 /usr/lib 的
#       libstdc++.so.6 较旧。conda 的 libstdcxx-ng 已包含 3.4.29，只需优先加载。
# 冲突：无。conda 的 libstdc++ 与 SparseDiff/rdkit/torch 等兼容；若需可再升级：
#       conda update -c conda-forge libstdcxx-ng
#
# 用法：
#   source scripts/fix_libstdc_conda.sh          # 当前 shell 生效
#   bash scripts/fix_libstdc_conda.sh --install  # 写入 conda 环境 activate.d，永久生效

set -e
cd "$(dirname "$0")/.."

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /data2/lyh/miniconda3/etc/profile.d/conda.sh ]; then
    source /data2/lyh/miniconda3/etc/profile.d/conda.sh
fi
conda activate sparse 2>/dev/null || { echo "Error: conda env 'sparse' not found"; exit 1; }

# 1) 可选：升级 libstdcxx-ng（当前 15.2 已含 3.4.29，通常不必）
if [ "$1" = "--upgrade" ]; then
    echo "Upgrading libstdcxx-ng (optional)..."
    conda update -c conda-forge libstdcxx-ng -y || true
    shift
fi

# 2) 当前 shell：优先用 conda 的 lib
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH prepended with: $CONDA_PREFIX/lib"

# 3) 可选：写入 activate.d，以后每次 conda activate sparse 自动生效
if [ "$1" = "--install" ]; then
    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
    cat > "$CONDA_PREFIX/etc/conda/activate.d/libstdc-fix.sh" << 'EOF'
# 优先使用 conda 的 libstdc++，避免 GLIBCXX_3.4.29 not found
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
EOF
    echo "Installed: $CONDA_PREFIX/etc/conda/activate.d/libstdc-fix.sh"
    echo "Re-run 'conda activate sparse' to apply permanently."
fi

# 快速自检（可选）
if [ "$1" != "--install" ] && [ "$1" != "--upgrade" ]; then
    echo "Quick check: python -c 'from rdkit import Chem' ..."
    python -c 'from rdkit import Chem; print("rdkit OK")' 2>/dev/null || echo "rdkit import failed (run with --install if needed)."
fi
