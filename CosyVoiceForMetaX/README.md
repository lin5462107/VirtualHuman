# CosyVoice for Metax GPU Platform

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
conda create -n CosyVoice python=3.10
conda activate CosyVoice
cd ./CosyVoiceForMetaX
```

#### Download PyTorch and ONNXRuntime from MetaX Developer Center
**Note** Please download the version that matches the Driver, such as `2.33.x.x`.

PyTorch: [link](https://developer.metax-tech.com/softnova/category?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=pytorch&ai_label=Pytorch)

ONNXRuntime: [link](https://developer.metax-tech.com/softnova/category?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=onnxruntime&ai_label=ONNXRuntime)


You will receive tar archives. After extracting them, navigate to the `wheel` directory and install using `pip`.
``` shell
# install PyTorch
tar -xvf maca-pytorch2.4-py310-2.33.0.6-x86_64.tar.xz
cd 2.33.0.6/wheel/
pip install ./*.whl

# install ONNXRuntime
tar -xvf maca-onnxruntime-py310-3.1.0.5-linux-x86_64.tar.xz
cd maca-onnxruntime-3.1.0.5/wheel/
pip install ./*.whl

```

#### Install Python Packages
``` shell
pip install -r requirements.txt
sudo apt-get install sox libsox-dev
```

#### Clone the Repo
``` shell
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# If you failed to clone the submodule due to network failures, please run the following command until success
cd CosyVoice
git submodule update --init --recursive
```

#### Model Download
``` python
# SDK model download
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

``` sh
# Git model download, please make sure Git LFS is installed
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models/CosyVoice-300M-SFT
git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git pretrained_models/CosyVoice-300M-Instruct
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

Optionally, you can unzip `ttsfrd` resource and install `ttsfrd` package for better text normalization performance.

Notice that this step is not necessary. If you do not install `ttsfrd` package, we will use wetext by default.

``` sh
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

#### Start web demo

You can use our web demo page to get familiar with CosyVoice quickly.

Please see the demo website for details.

``` python
# change iic/CosyVoice-300M-SFT for sft inference, or iic/CosyVoice-300M-Instruct for instruct inference
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```

#### Other References
Follow [CosyVoice](https://github.com/FunAudioLLM/CosyVoice.git).