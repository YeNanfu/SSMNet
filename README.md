# Efficient Spatiotemporal-Structural Masking for Dynamic Human Activity Recognition with Optimized Computation

## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets
```
WISDM, UniMiB, PAMAOP2
```
### Environments

Environment details used for the main experiments. Every main experiment is conducted on a single RTX 3090 GPU.

```
Environment:
	Python: 3.8.18
	PyTorch: 1.12.1 
	Torchvision: 0.13.1
```

### File Description

models.resnet is the baseline network；

models.dacd_resnet is the proposed DACDNet；

models.utils_dacd is the necessary components of the DACDNet；

all python files in utils is the components for training such as loss function, optimizer and etc
