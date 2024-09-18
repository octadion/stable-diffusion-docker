# Stage 1: Base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base

ARG TORCH_VERSION=2.0.1
ARG XFORMERS_VERSION=0.0.22
ARG WEBUI_VERSION=v1.7.0
ARG DREAMBOOTH_COMMIT=cf086c536b141fc522ff11f6cffc8b7b12da04b9
ARG KOHYA_VERSION=v22.6.1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/London \
    PYTHONUNBUFFERED=1 \
    SHELL=/bin/bash

# Create workspace working directory
WORKDIR /

# Install Ubuntu packages
RUN apt update && \
    apt -y upgrade && \
    apt install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        python3.10-venv \
        python3-pip \
        python3-tk \
        python3-dev \
        nodejs \
        npm \
        bash \
        dos2unix \
        git \
        git-lfs \
        ncdu \
        nginx \
        net-tools \
        inetutils-ping \
        openssh-server \
        libglib2.0-0 \
        libsm6 \
        libgl1 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        wget \
        curl \
        psmisc \
        rsync \
        vim \
        zip \
        unzip \
        p7zip-full \
        htop \
        screen \
        tmux \
        pkg-config \
        plocate \
        libcairo2-dev \
        libgoogle-perftools4 \
        libtcmalloc-minimal4 \
        apt-transport-https \
        ca-certificates && \
    update-ca-certificates && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Set Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install Torch, xformers and tensorrt
RUN pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir xformers==${XFORMERS_VERSION} tensorrt

# Stage 2: Install applications
FROM base as setup

RUN mkdir -p /sd-models

# Add SDXL models and VAE
# These need to already have been downloaded:
#   wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
#   wget https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors
#   wget https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors
COPY sd_xl_base_1.0.safetensors /sd-models/sd_xl_base_1.0.safetensors
COPY sd_xl_refiner_1.0.safetensors /sd-models/sd_xl_refiner_1.0.safetensors
COPY sdxl_vae.safetensors /sd-models/sdxl_vae.safetensors

# Clone the git repo of the Stable Diffusion Web UI by Automatic1111
# and set version
WORKDIR /
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd /stable-diffusion-webui && \
    git checkout tags/${WEBUI_VERSION}

WORKDIR /stable-diffusion-webui
RUN python3 -m venv --system-site-packages /venv && \
    source /venv/bin/activate && \
    pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir xformers==${XFORMERS_VERSION} && \
    pip3 install tensorflow[and-cuda] && \
    deactivate

# Install the dependencies for the Automatic1111 Stable Diffusion Web UI
COPY a1111/requirements.txt a1111/requirements_versions.txt ./
COPY a1111/cache-sd-model.py a1111/install-automatic.py ./
RUN source /venv/bin/activate && \
    python3 -m install-automatic --skip-torch-cuda-test && \
    deactivate

# Cache the Stable Diffusion Models
# SDXL models result in OOM kills with 8GB system memory, need 30GB+ to cache these
RUN source /venv/bin/activate && \
    python3 cache-sd-model.py --no-half-vae --no-half --xformers --use-cpu=all --ckpt /sd-models/sd_xl_base_1.0.safetensors && \
    python3 cache-sd-model.py --no-half-vae --no-half --xformers --use-cpu=all --ckpt /sd-models/sd_xl_refiner_1.0.safetensors && \
    deactivate

# Clone the Automatic1111 Extensions
RUN git clone https://github.com/d8ahazard/sd_dreambooth_extension.git extensions/sd_dreambooth_extension && \
    git clone --depth=1 https://github.com/deforum-art/sd-webui-deforum.git extensions/deforum && \
    git clone --depth=1 https://github.com/Mikubill/sd-webui-controlnet.git extensions/sd-webui-controlnet && \
    git clone --depth=1 https://github.com/ashleykleynhans/a1111-sd-webui-locon.git extensions/a1111-sd-webui-locon && \
    git clone --depth=1 https://github.com/Gourieff/sd-webui-reactor.git extensions/sd-webui-reactor && \
    git clone --depth=1 https://github.com/zanllp/sd-webui-infinite-image-browsing.git extensions/infinite-image-browsing && \
    git clone --depth=1 https://github.com/Uminosachi/sd-webui-inpaint-anything.git extensions/inpaint-anything && \
    git clone --depth=1 https://github.com/Bing-su/adetailer.git extensions/adetailer && \
    git clone --depth=1 https://github.com/civitai/sd_civitai_extension.git extensions/sd_civitai_extension && \
    git clone --depth=1 https://github.com/BlafKing/sd-civitai-browser-plus.git extensions/sd-civitai-browser-plus

# Install dependencies for Deforum, ControlNet, ReActor, Infinite Image Browsing,
# After Detailer, and CivitAI Browser+ extensions
RUN source /venv/bin/activate && \
    cd /stable-diffusion-webui/extensions/deforum && \
    pip3 install -r requirements.txt && \
    cd /stable-diffusion-webui/extensions/sd-webui-controlnet && \
    pip3 install -r requirements.txt && \
    cd /stable-diffusion-webui/extensions/sd-webui-reactor && \
    pip3 install -r requirements.txt && \
    pip3 install onnxruntime-gpu && \
    cd /stable-diffusion-webui/extensions/infinite-image-browsing && \
    pip3 install -r requirements.txt && \
    cd /stable-diffusion-webui/extensions/adetailer && \
    python3 -m install && \
    cd /stable-diffusion-webui/extensions/sd_civitai_extension && \
    pip3 install -r requirements.txt && \
    deactivate

# Install dependencies for inpaint anything extension
RUN source /venv/bin/activate && \
    pip3 install segment_anything lama_cleaner && \
    deactivate

# Install dependencies for Civitai Browser+ extension
RUN source /venv/bin/activate && \
    cd /stable-diffusion-webui/extensions/sd-civitai-browser-plus && \
    pip3 install send2trash ZipUnicode fake-useragent && \
    deactivate

# Set Dreambooth extension version
WORKDIR /stable-diffusion-webui/extensions/sd_dreambooth_extension
RUN git checkout main && \
    git reset ${DREAMBOOTH_COMMIT} --hard

# Install the dependencies for the Dreambooth extension
WORKDIR /stable-diffusion-webui
COPY a1111/requirements_dreambooth.txt ./requirements.txt
RUN source /venv/bin/activate && \
    cd /stable-diffusion-webui/extensions/sd_dreambooth_extension && \
    pip3 install -r requirements.txt && \
    deactivate

# Add inswapper model for the ReActor extension
RUN mkdir -p /stable-diffusion-webui/models/insightface && \
    cd /stable-diffusion-webui/models/insightface && \
    wget https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx

# Configure ReActor to use the GPU instead of the CPU
RUN echo "CUDA" > /stable-diffusion-webui/extensions/sd-webui-reactor/last_device.txt

# Fix Tensorboard
RUN source /venv/bin/activate && \
    pip3 uninstall -y tensorboard tb-nightly && \
    pip3 install tensorboard==2.15.2 tensorflow && \
    pip3 cache purge && \
    deactivate

# Install Kohya_ss
RUN git clone https://github.com/bmaltais/kohya_ss.git /kohya_ss
WORKDIR /kohya_ss
COPY kohya_ss/requirements* ./
RUN git checkout ${KOHYA_VERSION} && \
    python3 -m venv --system-site-packages venv && \
    source venv/bin/activate && \
    pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir xformers==${XFORMERS_VERSION} \
        bitsandbytes==0.41.1 \
        tensorboard==2.14.1 \
        tensorflow==2.14.0 \
        wheel \
        scipy \
        tensorrt && \
    pip3 install -r requirements.txt && \
    pip3 install . && \
    pip3 cache purge && \
    deactivate

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /ComfyUI
WORKDIR /ComfyUI
RUN python3 -m venv --system-site-packages venv && \
    source venv/bin/activate && \
    pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir xformers==${XFORMERS_VERSION} && \
    pip3 install -r requirements.txt && \
    deactivate

RUN mkdir -p /ComfyUI/models/controlnet \
    /ComfyUI/models/controlnet/SDXL/instantid \
    /ComfyUI/models/facedetection \
    /ComfyUI/models/facerestore_models \
    /ComfyUI/models/ipadapter \
    /ComfyUI/models/insightface \
    /ComfyUI/models/instantid \
    /ComfyUI/models/loras/ipadapter \
    /ComfyUI/models/sams \
    /ComfyUI/models/ultralytics/bbox \
    /ComfyUI/models/ultralytics/segm

# Download IP-Adapter models
WORKDIR /ComfyUI/models/ipadapter
RUN wget -O ip-adapter-faceid-plusv2_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin && \
    wget -O ip-adapter-faceid-plusv2_sdxl.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin && \
    wget -O ip-adapter-faceid-portrait-v11_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait-v11_sd15.bin && \
    wget -O ip-adapter-faceid-portrait_sdxl.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl.bin && \
    wget -O ip-adapter-faceid-portrait_sdxl_unnorm.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl_unnorm.bin && \
    wget -O ip-adapter-faceid_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin && \
    wget -O ip-adapter-faceid_sdxl.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin && \
    wget -O ip-adapter-full-face_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors && \
    wget -O ip-adapter-plus-face_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors && \
    wget -O ip-adapter-plus-face_sdxl_vit-h.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors && \
    wget -O ip-adapter-plus_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors && \
    wget -O ip-adapter_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors && \
    wget -O ip-adapter_sd15_light_v11.bin https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin && \
    wget -O ip-adapter-plus_sdxl_vit-h.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors && \
    wget -O ip-adapter_sdxl.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors && \
    wget -O ip-adapter_sdxl_vit-h.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors && \
    wget -O ip_plus_composition_sdxl.safetensors https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sdxl.safetensors && \
    wget -O ip_plus_composition_sd15.safetensors https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sd15.safetensors && \
    wget -O ip-adapter_sd15_vit-G.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors

# Download Insightface models
WORKDIR /ComfyUI/models/insightface
RUN wget -O inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx && \
    wget -O 1k3d68.onnx https://huggingface.co/menglaoda/_insightface/resolve/main/1k3d68.onnx && \
    wget -O 2d106det.onnx https://huggingface.co/menglaoda/_insightface/resolve/main/2d106det.onnx && \
    wget -O det_10g.onnx https://huggingface.co/menglaoda/_insightface/resolve/main/det_10g.onnx && \
    wget -O genderage.onnx https://huggingface.co/menglaoda/_insightface/resolve/main/genderage.onnx && \
    wget -O w600k_r50.onnx https://huggingface.co/menglaoda/_insightface/resolve/main/w600k_r50.onnx

# Download InstantID model
WORKDIR /ComfyUI/models/instantid
RUN wget -O ip-adapter.bin https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin

# Download Loras/IP-Adapter models
WORKDIR /ComfyUI/models/loras/ipadapter
RUN wget -O ip-adapter-faceid-plusv2_sd15_lora.safetensors https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors && \
    wget -O ip-adapter-faceid-plusv2_sdxl_lora.safetensors https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors && \
    wget -O ip-adapter-faceid_sd15_lora.safetensors https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors && \
    wget -O ip-adapter-faceid_sdxl_lora.safetensors https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl_lora.safetensors

# Download Sams models
WORKDIR /ComfyUI/models/sams
RUN wget -O sam_vit_b_01ec64.pth https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth

# Download Ultralytics/BBox models
WORKDIR /ComfyUI/models/ultralytics/bbox
RUN wget -O face_yolov8m.pt https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt && \
    wget -O hand_yolov8s.pt https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt

# Download Ultralytics/Segm models
WORKDIR /ComfyUI/models/ultralytics/segm
RUN wget -O person_yolov8m-seg.pt https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt

# Download ControlNet models
WORKDIR /ComfyUI/models/controlnet
RUN wget -O control_sd15_inpaint_depth_hand_fp16.safetensors https://huggingface.co/hr16/ControlNet-HandRefiner-pruned/blob/main/control_sd15_inpaint_depth_hand_fp16.safetensors && \
    wget -O control_v11e_sd15_ip2p_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors && \
    wget -O control_v11f1e_sd15_tile_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/blob/main/control_v11f1e_sd15_tile_fp16.safetensors && \
    wget -O control_v11f1p_sd15_depth_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/blob/main/control_v11f1p_sd15_depth_fp16.safetensors && \
    wget -O control_v11p_sd15_canny_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors && \
    wget -O control_v11p_sd15_lineart_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors && \
    wget -O control_v11p_sd15_mlsd_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors && \
    wget -O control_v11p_sd15_normalbae_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/blob/main/control_v11p_sd15_normalbae_fp16.safetensors && \
    wget -O control_v11p_sd15_openpose_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/blob/main/control_v11p_sd15_openpose_fp16.safetensors && \
    wget -O control_v11p_sd15_scribble_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/blob/main/control_v11p_sd15_scribble_fp16.safetensors && \
    wget -O control_v11p_sd15_softedge_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/blob/main/control_v11p_sd15_softedge_fp16.safetensors && \
    wget -O control_v11p_sd15s2_lineart_anime_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/blob/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors && \
    wget -O control_v11u_sd15_tile_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/blob/main/control_v11u_sd15_tile_fp16.safetensors

# Download ControlNet SDXL instantid model
WORKDIR /ComfyUI/models/controlnet/SDXL/instantid
RUN wget -O diffusion_pytorch_model.safetensors https://huggingface.co/lllyasviel/sd-controlnet-canny/blob/main/diffusion_pytorch_model.safetensors

# Download Face detection models
WORKDIR /ComfyUI/models/facedetection
RUN wget -O detection_Resnet50_Final.pth https://huggingface.co/gmk123/GFPGAN/blob/main/detection_Resnet50_Final.pth && \
    wget -O parsing_parsenet.pth https://huggingface.co/gmk123/GFPGAN/blob/main/parsing_parsenet.pth

# Download Face restore models
WORKDIR /ComfyUI/models/facerestore_models
RUN wget -O GFPGANv1.3.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth && \
    wget -O GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth && \
    wget -O GPEN-BFR-1024.onnx https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/facerestore_models/GPEN-BFR-1024.onnx && \
    wget -O GPEN-BFR-2048.onnx https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/facerestore_models/GPEN-BFR-2048.onnx && \
    wget -O GPEN-BFR-512.onnx https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/facerestore_models/GPEN-BFR-512.onnx && \
    wget -O codeformer-v0.1.0.pth https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/facerestore_models/codeformer-v0.1.0.pth

# Install ComfyUI Custom Nodes and Additional Packages

# Clone repositories
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager && \
    git clone https://github.com/Gourieff/comfyui-reactor-node.git custom_nodes/comfyui-reactor-node && \
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git custom_nodes/comfyui-animatediff-evolved && \
    git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git custom_nodes/comfyui-advanced-controlnet && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git custom_nodes/comfyui-videohelpersuite && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git custom_nodes/comfyui-impact-pack && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git custom_nodes/comfyui_controlnet_aux && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git custom_nodes/comfyui-frame-interpolation && \
    git clone https://github.com/time-river/ComfyUI-CLIPSeg.git custom_nodes/comfyui-clipseg && \
    git clone https://github.com/jags111/efficiency-nodes-comfyui.git custom_nodes/efficiency-nodes-comfyui && \
    git clone https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes.git custom_nodes/derfuu_comfyui_moddednodes && \
    git clone https://github.com/WASasquatch/was-node-suite-comfyui.git custom_nodes/was-node-suite-comfyui && \
    git clone https://github.com/SLAPaper/ComfyUI-Image-Selector.git custom_nodes/comfyui-image-selector && \
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git custom_nodes/comfyui-custom-scripts && \
    git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git custom_nodes/comfyui-ultimate-sd-upscale && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git custom_nodes/comfyui-ipadapter-plus && \
    git clone https://github.com/sipherxyz/comfyui-art-venture.git custom_nodes/comfyui-art-venture && \
    git clone https://github.com/wallish77/wlsh_nodes.git custom_nodes/wlsh_nodes && \
    git clone https://github.com/M1kep/ComfyLiterals.git custom_nodes/comfy_literals && \
    git clone https://github.com/rgthree/rgthree-comfy.git custom_nodes/rgthree-comfy && \
    git clone https://github.com/storyicon/comfyui_segment_anything.git custom_nodes/comfyui_segment_anything && \
    git clone https://github.com/chflame163/ComfyUI_LayerStyle.git custom_nodes/comfyui_layerstyle && \
    git clone https://github.com/edenartlab/eden_comfy_pipelines.git custom_nodes/eden_comfy_pipelines && \
    git clone https://github.com/Excidos/ComfyUI-Documents.git custom_nodes/comfyui-documents && \
    git clone https://github.com/un-seen/comfyui_segment_anything_plus.git custom_nodes/comfyui_segment_anything_plus && \
    git clone https://github.com/neverbiasu/ComfyUI-SAM2.git custom_nodes/comfyui-sam2

# Activate virtual environment
RUN source /ComfyUI/venv/bin/activate

# Install dependencies for specific nodes
RUN cd custom_nodes/ComfyUI-Manager && pip3 install -r requirements.txt && cd ../../

RUN cd custom_nodes/comfyui-reactor-node && python3 install.py && cd ../../

RUN cd custom_nodes/comfyui-animatediff-evolved && pip3 install . && cd ../../

RUN cd custom_nodes/comfyui-advanced-controlnet && pip3 install . && cd ../../

RUN cd custom_nodes/comfyui-videohelpersuite && pip3 install -r requirements.txt && pip3 install . && cd ../../

RUN cd custom_nodes/comfyui-impact-pack && pip3 install -r requirements.txt && python3 install.py && cd ../../

RUN cd custom_nodes/comfyui_controlnet_aux && pip3 install -r requirements.txt && cd ../../

RUN cd custom_nodes/comfyui-frame-interpolation && python3 install.py && cd ../../

# RUN cd custom_nodes/comfyui-clipseg && pip3 install -r requirements.txt && cd ../../

RUN cd custom_nodes/efficiency-nodes-comfyui && pip3 install -r requirements.txt && pip3 install . && cd ../../

RUN cd custom_nodes/derfuu_comfyui_moddednodes && pip3 install . && cd ../../

RUN cd custom_nodes/was-node-suite-comfyui && pip3 install -r requirements.txt && cd ../../

RUN cd custom_nodes/comfyui-art-venture && pip3 install -r requirements.txt && cd ../../

RUN cd custom_nodes/wlsh_nodes && pip3 install -r requirements.txt && cd ../../

RUN cd custom_nodes/comfyui_segment_anything && pip3 install -r requirements.txt && python3 install.py && cd ../../

RUN cd custom_nodes/comfyui_layerstyle && \
    ../../../python_embeded/python.exe -s -m pip install ./whl/docopt-0.6.2-py2.py3-none-any.whl && \
    ../../../python_embeded/python.exe -s -m pip install ./whl/hydra_core-1.3.2-py3-none-any.whl && \
    ../../../python_embeded/python.exe -s -m pip install -r requirements.txt && \
    ./repair_dependency.bat && cd ../../

RUN cd custom_nodes/eden_comfy_pipelines && pip3 install -r requirements.txt && cd ../../

RUN cd custom_nodes/comfyui-documents && pip3 install -r requirements.txt && cd ../../

RUN cd custom_nodes/comfyui_segment_anything_plus && pip3 install -r requirements.txt && python3 install.py && cd ../../

RUN cd custom_nodes/comfyui-sam2 && pip3 install -r requirements.txt && cd ../../

# Clean up cache and deactivate virtual environment
RUN pip3 cache purge && deactivate


# Install Application Manager
WORKDIR /
RUN git clone https://github.com/oddomatik/app-manager.git /app-manager && \
    cd /app-manager && \
    npm install

# Install Jupyter, gdown and OhMyRunPod
RUN pip3 install -U --no-cache-dir jupyterlab \
        jupyterlab_widgets \
        ipykernel \
        ipywidgets \
        gdown \
        OhMyRunPod

# Install RunPod File Uploader
RUN curl -sSL https://github.com/kodxana/RunPod-FilleUploader/raw/main/scripts/installer.sh -o installer.sh && \
    chmod +x installer.sh && \
    ./installer.sh

# Install rclone
RUN curl https://rclone.org/install.sh | bash

# Install runpodctl
RUN wget https://github.com/runpod/runpodctl/releases/download/v1.13.0/runpodctl-linux-amd64 -O runpodctl && \
    chmod a+x runpodctl && \
    mv runpodctl /usr/local/bin

# Install croc
RUN curl https://getcroc.schollz.com | bash

# Install speedtest CLI
RUN curl -s https://packagecloud.io/install/repositories/ookla/speedtest-cli/script.deb.sh | bash && \
    apt install speedtest

# Install CivitAI Model Downloader
RUN git clone --depth=1 https://github.com/ashleykleynhans/civitai-downloader.git && \
    mv civitai-downloader/download.py /usr/local/bin/download-model && \
    chmod +x /usr/local/bin/download-model

# Copy Stable Diffusion Web UI config files
COPY a1111/relauncher.py a1111/webui-user.sh a1111/config.json a1111/ui-config.json /stable-diffusion-webui/

# ADD SDXL styles.csv
ADD https://raw.githubusercontent.com/Douleb/SDXL-750-Styles-GPT4-/main/styles.csv /stable-diffusion-webui/styles.csv

# Copy ComfyUI Extra Model Paths (to share models with A1111)
COPY comfyui/extra_model_paths.yaml /ComfyUI/

# Remove existing SSH host keys
RUN rm -f /etc/ssh/ssh_host_*

# NGINX Proxy
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/502.html /usr/share/nginx/html/502.html
COPY nginx/README.md /usr/share/nginx/html/README.md

# Set template version
ENV TEMPLATE_VERSION=3.12.5

# Copy the scripts
WORKDIR /
COPY --chmod=755 scripts/* ./

# Copy the accelerate configuration
COPY kohya_ss/accelerate.yaml ./

# Start the container
SHELL ["/bin/bash", "--login", "-c"]
CMD [ "/start.sh" ]
