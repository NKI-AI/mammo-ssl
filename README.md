# Self-supervised learning of NKI mammograms using VISSL

We use vissl release v0.1.6, which is only supported for torch <= 1.9.1. This requires a specific installation for torch and cuda. Steps to initialize a working conda environment:

```
conda create -n visslmammo python=3.9

conda activate visslmammo

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py39_cu111_pyt191/download.html

pip install -r requirements.txt

pip uninstall -y classy_vision

pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d

pip uninstall -y fairscale

pip install fairscale@https://github.com/facebookresearch/fairscale/tarball/df7db85cef7f9c30a5b821007754b96eb1f977b6

pip install -e .[dev]
```
