# SliceSamp: A Promising Downsampling Alternative for Retaining Information in Neural Network	
An alternative downsampling method that is lightweight, efficient, and promising

**[WHU Research](https://github.com/OyamingO/SliceSamp/)**

Lianlian He, [Ming Wang*](https://github.com/OyamingO)

[[`Paper`]()] [[`Project`](https://github.com/OyamingO)] [[`Demo`](#Citing-SliceSamp)] [[`Dataset`](https://doi.org/10.57760/sciencedb.j00104.00103)]  [[`BibTeX`](#Citing-SliceSamp)]


### 🔥: SliceSamp design
<img src="assets/SliceSamp.jpg?raw=true" width="66%" />

### 🚀: SliceUpsamp design
<img src="assets/SliceUpsamp.jpg?raw=true" width="66%" />

## Installation

The code requires `python>=3.7`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib
```

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Citing SliceSamp

If you use SliceSamp in your research, please use the following BibTeX entry.

```
@article{SliceSamp,
  title={SliceSamp: A Promising Downsampling Alternative for Retaining Information in Neural Network},
  author={Lianlian He, Ming Wang},
  journal={Applied Sciences},
  year={2023}
}
```
