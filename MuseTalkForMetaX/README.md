# MuseTalk for Metax GPU Platform

## About

Since some AI libraries are associated with GPU platforms, on MetaX's GPU platform, these libraries need to be recompiled for compatibility. MetaX provides a variety of AI libraries that have already been compiled MetaX's GPU platform, which can be downloaded and installed from the MetaX developer community (https://developer.metax-tech.com/).

For this project, follow the steps below for compatibility.

## Installation

Recommended runtime environment:

OS: Ubuntu 22.04

Python: 3.10

PyTorch >= 2.0

GPU: MetaX C500

Driver/SDK Version: 2.33.0.6 or higger

RAM: >= 128GB

### Install MetaX C500 Driver and SDK
1. Go to the [MetaX Developer Center](https://sw-developer.metax-tech.com/member.php?mod=register) to register an account.

2. Download MetaX C500 [Driver](https://developer.metax-tech.com/softnova/download?package_kind=Driver&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85) and [SDK](https://developer.metax-tech.com/softnova/download?package_kind=SDK&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85), version: 2.33.0.6 or higger. Please download the local install version.

3. Follow the instructions on the webpage to complete the installation.

4. Update `.bashrc` file.
    
    ```
    vi ~/.bashrc
    ```
    Add the following environment variables
    ```shell
    export MACA_PATH=/opt/maca
    export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
    export PATH=${MACA_PATH}/bin:${MACA_PATH}/tools/cu-bridge/bin:${MACA_PATH}/tools/cu-bridge/tools:${MACA_CLANG_PATH}:${PATH}
    export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH}
    export MXLOG_LEVEL=err
    ```
    Update environment variables
    ```
    source ~/.bashrc
    ```
5. Add current user to video group.

    ```shell
    sudo usermod -aG video $USER
    newgrp video
    ```

6. Reboot your server.

7. You can use the `mx-smi` command to check GPU information.

### Using Conda(Suggested)
#### Create python environment
``` shell
# create conda environment
conda create -n MuseTalk python=3.10
conda activate MuseTalk
cd ./MuseTalkForMetaX
```

#### Download PyTorch, Tensorflow2 and MMCV from MetaX Developer Center
**Note** Please download the version that matches the Driver, such as `2.33.x.x`.

PyTorch: [link](https://developer.metax-tech.com/softnova/category?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=pytorch&ai_label=Pytorch)

Tensorflow2: [link](https://developer.metax-tech.com/softnova/category?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=tensorflow2&ai_label=TensorFlow2)

MMCV: [link](https://developer.metax-tech.com/softnova/category?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=mmcv&ai_label=MMCV)

You will receive tar archives. After extracting them, navigate to the `wheel` directory and install using `pip`.
``` shell
# install PyTorch
tar -xvf maca-pytorch2.4-py310-2.33.0.6-x86_64.tar.xz
cd 2.33.0.6/wheel/
pip install ./*.whl

# install Tensorflow2
tar -xvf maca-tensorflow2-2.13.1-py310-3.2.1.5-linux-x86_64.tar.xz
cd maca-tensorflow2-3.2.1.5/wheel/
pip install ./*.whl

# install MMCV
# install mmengine
pip install --no-cache-dir -U openmim
mim install mmengine
# install mmcv
tar -xvf maca-mmcv-py310-3.1.0.5-linux-x86_64.tar.xz
cd maca-mmcv-3.1.0.5
pip install ./*.whl
# install mmdet
# if there is a version mismatch error in runtime, remove the checking version code.
git clone https://github.com/open-mmlab/mmdetection.git -b v3.1.0 
cd mmdetection 
pip install -r requirements.txt
pip install -e .  
# install mmpose
git clone https://github.com/open-mmlab/mmpose.git -b v1.2.0
cd mmpose
cp ../../third_party/mmpose/requirements/* ./requirements
pip install -r requirements.txt
pip install -e .  
```

#### Install python packages
``` shell
cd ./MuseTalk
pip install -r requirements.txt
```

#### Setup FFmpeg
```bash
sudo apt-get install ffmpeg
```

#### Clone the Repo
``` shell
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
```

#### Download Weights
``` shell
sh ./download_weights.sh
```

#### Inference
``` shell
# MuseTalk 1.5 (Recommended)
sh inference.sh v1.5 normal

# MuseTalk 1.0
sh inference.sh v1.0 normal
``` 

## Other References
Follow [MuseTalk](https://github.com/TMElyralab/MuseTalk.git).