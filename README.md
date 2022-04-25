# ai4cop-health-cams

AI4Copernicus health service: downscaling CAMS model output with deep learning.

**NB**: A recent GPU is required to train a SRGAN model. Training on the CPU will likely take a long time. Inference is possible on the CPU, but may take a long time - depending on input (sample) size.

## How to use

0. Create the Docker image and run it (or deploy to Kubernetes):

```shell
$> git clone https://github.com/mishooax/ai4cop-health-cams.git
$> cd ai4cop-health-cams
$> docker build --tag ai4cop-health-cams:v1 -f docker/Dockerfile .
# optional: add tag, push it to your private Docker repo
# run the container in interactive mode or edit the CMD in the Dockerfile to e.g. start a Jupyter server
$> docker exec -it ai4cop-health-cams bash
```

1. Download CAMS model output data from the Copernicus ADS. You will have to register for an user account first. Instructions on how to do that are here:

2. Pre-process the input (low-res and hi-res) data to netCDF4 format. The hi-res data should be 8x the resolution of the coarse input. The service has been tested with an input dataset obtained by coarsening the high-res output by 8x. You can, of course, use any input / output data sets as you consider fit. Make sure to set the correct input paths in the configuration YAML file. See `src/config/config.yaml` for an example.

3. Pre-training a generator model (you can use this on its own or couple it to a GAN in step 4):

```shell
$ ai4cop-cams-pretrain --help
usage: ai4cop-cams-pretrain [-h] --model {srgan} --config CONFIG

optional arguments:
  -h, --help       show this help message and exit

required arguments:
  --model {srgan}  Super-resolution model
  --config CONFIG  Model configuration file (YAML)
```

4. Training:

```shell
$ ai4cop-cams-train --help
usage: ai4cop-cams-train [-h] --model {srgan,unet,xnet,swin} --config CONFIG [--pretrained-generator]

optional arguments:
  -h, --help            show this help message and exit

required arguments:
  --model {srgan,unet,xnet,swin}
                        Super-resolution model
  --config CONFIG       Model configuration file (YAML)

optional arguments:
  --pretrained-generator
```

5. Inference (prediction):

```shell
$ ai4cop-cams-predict --help
usage: ai4cop-cams-predict [-h] --model {srgan} --config CONFIG

optional arguments:
  -h, --help       show this help message and exit

required arguments:
  --model {srgan}  Pre-trained super-resolution model
  --config CONFIG  Model configuration file (YAML)
```

Questions? Please contact info@ai4copernicus-project.eu.
