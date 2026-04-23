#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"


export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_SAMPLES_INC="$REPO_DIR/external/cuda-samples/Common"
mkdir -p "$REPO_DIR/tmp/build"
export TMPDIR="${TMPDIR:-$REPO_DIR/tmp/build}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

python -m pip install --upgrade pip

# Core stack aligned with the working vistracker_meshisect environment.
conda install -y -c pytorch -c nvidia \
  pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8
conda install -y -c conda-forge igl
conda install -y https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py38_cu118_pyt200.tar.bz2

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# Other Python deps used by VisTracker.
python -m pip install \
  chumpy==0.70 \
  imageio==2.35.1 \
  imageio-ffmpeg==0.5.1 \
  jupyterlab==3.6.8 \
  joblib \
  matplotlib \
  notebook==6.3.0 \
  numba==0.55.1 \
  numexpr \
  numpy \
  open3d==0.14.1 \
  opencv-python \
  Pillow \
  protobuf==3.20.1 \
  PyYAML \
  scikit-image==0.21.0 \
  scikit-learn \
  scipy==1.10.1 \
  tensorboard==2.5.0 \
  tqdm \
  trimesh==3.9.1 \
  yacs \
  zstd==1.5.0.4

# External packages: reuse existing clones if present.
mkdir -p external

if [ ! -d external/mesh ]; then
  git clone https://github.com/MPI-IS/mesh.git external/mesh
fi
git -C external/mesh checkout 49e70425cf373ec5269917012bda2944215c5ccd

if [ ! -d external/neural_renderer ]; then
  git clone https://github.com/daniilidis-group/neural_renderer.git external/neural_renderer
fi
git -C external/neural_renderer checkout b2a1e6ce16a54f94f26f86fe1dc3814637e15251

if [ ! -d external/cuda-samples ]; then
  git clone https://github.com/NVIDIA/cuda-samples.git external/cuda-samples
fi
git -C external/cuda-samples checkout a4526d52290b667cbc46b4a9830fbaad35be1ec2

if [ ! -d external/torch-mesh-isect ]; then
  git clone https://github.com/dingbang777/torch-mesh-isect.git external/torch-mesh-isect
fi


# neural_renderer compatibility for current torch headers
for f in \
  external/neural_renderer/neural_renderer/cuda/load_textures_cuda.cpp \
  external/neural_renderer/neural_renderer/cuda/rasterize_cuda.cpp \
  external/neural_renderer/neural_renderer/cuda/create_texture_image_cuda.cpp
do
  grep -q "define AT_CHECK TORCH_CHECK" "$f" || \
    perl -0pi -e 's@#include <torch/torch.h>\n@#include <torch/torch.h>\n\n#ifndef AT_CHECK\n#define AT_CHECK TORCH_CHECK\n#endif\n@' "$f"
done

python -m pip install ./external/mesh
python -m pip install ./external/neural_renderer

(cd external/torch-mesh-isect && python setup.py develop)


# Public demo resources from the README
mkdir -p downloads
curl -L -o downloads/models.zip https://datasets.d2.mpi-inf.mpg.de/cvpr23vistracker/models.zip
curl -L -o downloads/behave-splits.zip https://datasets.d2.mpi-inf.mpg.de/cvpr23vistracker/behave-splits.zip

mkdir -p experiments demo_data behave_release splits
unzip -oq downloads/models.zip -d experiments
unzip -oq downloads/behave-splits.zip -d splits

ln -sfn ../../../body_models/smpl/SMPL_MALE.pkl smplh_models/SMPL_male.pkl
ln -sfn ../../../body_models/smpl/SMPL_FEMALE.pkl smplh_models/SMPL_female.pkl
ln -sfn ../../../body_models/mano_v1_2/models/SMPLH_male.pkl smplh_models/SMPLH_male.pkl
ln -sfn ../../../body_models/mano_v1_2/models/SMPLH_female.pkl smplh_models/SMPLH_female.pkl


# Some code paths look for object templates under dirname(BEHAVE_PATH)/objects.
# For the demo layout BEHAVE_PATH=./demo_data, so add a local objects symlink.
ln -sfn behave_release/objects objects


# Quick verification.
python - <<'PY'
import torch
import detectron2
import pytorch3d
import open3d
import igl
from mesh_intersection.bvh_search_tree import BVH

print('torch', torch.__version__, 'torch_cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())
print('detectron2', detectron2.__version__)
print('pytorch3d', pytorch3d.__version__)
print('open3d', open3d.__version__)
print('igl', igl.__file__)

triangles = torch.tensor(
    [[
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.25, 0.25, -1.0], [0.25, 0.25, 1.0], [0.75, 0.75, 0.0]],
    ]],
    device='cuda'
)
print(BVH(max_collisions=8)(triangles)[0, :4].cpu())
PY
