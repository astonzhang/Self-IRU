# Self-Instantiated Recurrent Units with Dynamic Soft Recursion (Self-IRU)

This repository contains the PyTorch implementation of the Self-IRU model in the paper *Self-Instantiated Recurrent Units with Dynamic Soft Recursion* at NeurIPS 2021.


## Installation

One needs to install the following libraries

* [PyTorch](https://pytorch.org/) (e.g., v1.4.0)
* [Sequence-to-Sequence Toolkit](https://github.com/pytorch/fairseq)
* [CuPy](https://cupy.dev/)
* [Python Bindings to NVRTC](https://github.com/NVIDIA/pynvrtc)
* [SciPy](https://www.scipy.org/)


## Usage

The usage of this repository follows the [TCN](https://github.com/locuslab/TCN) repository (e.g., for polyphonic music tasks). To run the Self-IRU model, set `model = RNNModel(input_size, args.nhid, dropout=dropout, rnn_type='INFINITY', args=args)` in the `[TASK_NAME]_test.py` file, where `INFINITY` is the alias of the Self-IRU in our implementation. If you encounter `ModuleNotFoundError`, try `export PYTHONPATH="${PYTHONPATH}:."`.

## Citation

If you find this repository helpful, please cite our paper:

```
@article{zhang2021selfiru,
    title={Self-Instantiated Recurrent Units with Dynamic Soft Recursion
},
    author={Zhang, Aston and Tay, Yi and Shen, Yikang and Chan, Alvin and Zhang, Shuai},
    booktitle={Advances in neural information processing systems},
    year={2021}
}
```

