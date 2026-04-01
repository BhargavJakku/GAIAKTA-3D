# Phase 0: Install and Setup


 ## Install Ubuntu 24.04 LTS

 - wsl --install -d Ubuntu-24.04
 - wsl --set-default-version 2
 - wsl --update
 - wsl --shutdown
 - wsl -d Ubuntu-24.04
 - sudo apt update && sudo apt upgrade -y
 - sudo apt install -y curl git build-essential

 ## Get into Ubuntu

 - wsl -d Ubuntu-24.04
 - sudo apt update && sudo apt upgrade -y
 - sudo apt install -y curl git build-essential

 ## Move raw files into Windows folder
    - Move scanned files into Windows folder and get access to them from Ubuntu
    - 
 ## Install ffmpepg in Ubuntu
    - sudo apt install -y ffmpeg
    - Check version : ffmpeg -version
    - Create a Folder to store frames : mkdir -p /mnt/b/projects/GAIAKTA/frames/firefighter_memorial
    - downscale now in ffmpeg
    - Convert scanned files into frames : ffmpeg -i /mnt/b/projects/GAIAKTA/raw/firefighter_memorial/IMG_5814.MOV -qscale:v 2 vf "fps=2" /mnt/b/projects/GAIAKTA/frames/firefighter_memorial/vid1_%04d.jpg
    - Convert all the videos into frames and store them in the frames folder : is there a way to do it in batch mode? Yes: i=1; for f in /mnt/b/projects/GAIAKTA/raw/firefighter_memorial/*.MOV; do ffmpeg -i "$f" -qscale:v 2 -vf "fps=2,scale=1080:1920" /mnt/b/projects/GAIAKTA/frames_1080/firefighter_memorial/vid${i}_%04d.jpg; i=$((i+1)); done
    
 ## Install all colmap dependencies
    -sudo apt install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    libceres-dev

    - sudo apt install -y openimageio-tools
    - sudo apt install nvidia-cuda-toolkit 
    -cd ~
    git clone https://github.com/colmap/colmap.git
    cd colmap
    mkdir build && cd build
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES="120" -DGUI_ENABLED=OFF

 ## Install Ninja
    - sudo apt install -y ninja-build

 ## Run COLMAP
    - mkdir -p /mnt/b/projects/GAIAKTA/colmap/firefighter_memorial
    - colmap feature_extractor \
    --database_path /mnt/b/projects/GAIAKTA/colmap/firefighter_memorial/database.db \
    --image_path /mnt/b/projects/GAIAKTA/frames/firefighter_memorial \
    --FeatureExtraction.use_gpu 1
 
    -colmap exhaustive_matcher \
    --database_path /mnt/b/projects/GAIAKTA/colmap/firefighter_memorial/database.db \
    --FeatureMatching.use_gpu 1

    colmap model_analyzer \
    --path /mnt/b/projects/GAIAKTA/colmap/firefighter_memorial/sparse/0

 ## Activate virtual environment
    - cd /mnt/b/projects/GAIAKTA
    - python3 -m venv venv
    - source venv/bin/activate

## INstall pytorch
    -pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    - Verify : python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

## Install Nerfstudio
    - pip install nerfstudio
    - Verify : python3 -c "import nerfstudio; print(nerfstudio.__version__)"    

    - ns-process-data images \
    --data /mnt/b/projects/GAIAKTA/frames/firefighter_memorial \
    --output-dir /mnt/b/projects/GAIAKTA/ns_data/firefighter_memorial \
    --skip-colmap \
    --colmap-model-path /mnt/b/projects/GAIAKTA/colmap/firefighter_memorial/sparse/0

    - Add .wslconfig file in your user folder incase ubuntu got killed due to RAM issues
    [wsl]
    memory=12GB
    processors=8

    - ns-train splatfacto \
    --data /mnt/b/projects/GAIAKTA/ns_data/firefighter_memorial \
    --output-dir /mnt/b/projects/GAIAKTA/outputs \
    --max-num-iterations 30000

    - For 4 parallel CUDA jobs
    - MAX_JOBS=4 ns-train splatfacto \
    --output-dir /mnt/b/projects/GAIAKTA/outputs \
    --max-num-iterations 30000 \
    --pipeline.datamanager.cache-images cpu \
    nerfstudio-data \
    --data /mnt/b/projects/GAIAKTA/ns_data/firefighter_memorial \
    --downscale-factor 1