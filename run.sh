#!/usr/bin/env bash
#SBATCH --output=/share/sopio/slurm/%J.out
#SBATCH --error=/share/sopio/slurm/%J.err
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=1-0
#SBATCH --partition=2080
#SBATCH -J envsetup


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
squeue

# Activate a conda environment if necessary.
source /data/sopio/miniconda3/bin/activate
conda activate base

echo "Using conda env: $CONDA_PREFIX"
echo "Current dir: ${PWD}"

conda create -n instantsplatPP python=3.10.13 cmake=3.14.0 -y
conda activate instantsplatPP

# PyTorch + CUDA runtime (12.1)
conda install pytorch torchvision pytorch-cuda=12.4 cuda-toolkit=12.4 ninja -c pytorch -c nvidia -y
python -m pip install -U pip

pip install -r requirements.txt

python -m pip uninstall -y numpy opencv-python
python -m pip install --no-cache-dir "numpy<2" "opencv-python<4.12"

cat > constraints.txt << 'EOF'
numpy<2
opencv-python<4.12
EOF

# install from local repo (already under third_party/)
pip install -e /share/sopio/master_thesis/codebases/InstantSplatPP/third_party/map-anything -c constraints.txt

# enforce NumPy<2 (in case it was upgraded), then rebuild CUDA submodules
python -m pip install --no-cache-dir "numpy<2"

rm -rf /share/sopio/master_thesis/codebases/InstantSplatPP/submodules/simple-knn/build \
       /share/sopio/master_thesis/codebases/InstantSplatPP/submodules/diff-gaussian-rasterization/build \
       /share/sopio/master_thesis/codebases/InstantSplatPP/submodules/fused-ssim/build

export CUDA_HOME=/data/sopio/miniconda3/envs/instantsplatPP
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

pip install --no-build-isolation /share/sopio/master_thesis/codebases/InstantSplatPP/submodules/simple-knn
pip install --no-build-isolation /share/sopio/master_thesis/codebases/InstantSplatPP/submodules/diff-gaussian-rasterization
pip install --no-build-isolation /share/sopio/master_thesis/codebases/InstantSplatPP/submodules/fused-ssim