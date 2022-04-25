FROM nvidia/cuda:10.2-base-ubuntu18.04

LABEL maintainer="ECMWF <mihai.alexe@ecmwf.int>"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    bash \
    libx11-6 \
    vim \
 && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

RUN echo $0

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash ai4cop \
 && chown -R ai4cop:ai4cop /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER ai4cop

COPY --chown=ai4cop:ai4cop requirements.txt /tmp/requirements.txt
COPY --chown=ai4cop:ai4cop postBuild /usr/local/bin/postBuild.sh
RUN chmod u+x /usr/local/bin/postBuild.sh
COPY --chown=ai4cop:ai4cop docker/entrypoint.sh /usr/local/bin/
RUN chmod u+x /usr/local/bin/entrypoint.sh

ENV HOME=/home/ai4cop
RUN chmod 777 /home/ai4cop

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV MINICONDA_VERSION py38_4.11.0
ENV PATH=/home/ai4cop/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh

# make non-activate conda commands available
ENV CONDA_DIR $HOME/miniconda

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/sh and /bin/bash -- interative shells
RUN conda init bash

# copy source and notebook files
RUN mkdir src
COPY --chown=ai4cop:ai4cop src src

RUN mkdir notebooks
COPY --chown=ai4cop:ai4cop notebooks/*.ipynb notebooks/

# build the conda environment
ENV ENV_PREFIX /app/envs
RUN conda update --name base -c conda-forge conda && \
    conda install -c conda-forge --file /tmp/requirements.txt --force && \
    conda clean --all --yes && \
    rm /tmp/requirements.txt

# run the postBuild script to install the JupyterLab extensions
RUN source activate && \
    /usr/local/bin/postBuild.sh && \
    conda deactivate

# use an entrypoint script to ensure conda environment is properly activated at runtime
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

# Set the default command to /bin/bash
CMD ["/bin/bash"]
