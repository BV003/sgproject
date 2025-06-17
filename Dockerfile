FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# 安装 Conda
RUN apt-get update && apt-get install -y \
    wget git bzip2 build-essential \
    && wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:$PATH"
WORKDIR /workspace

# 拷贝项目代码（可选）
COPY . .

CMD ["bash"]