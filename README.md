# RTMDet

RTMDet is an implementation of the Real-Time Multi-task Detection (RTMDet) model, designed for efficient and accurate object detection tasks. This repository provides modular components for building, training, and evaluating RTMDet models using PyTorch.

## Installation

1. Clone the repository:
	 ```bash
	 git clone https://github.com/yourusername/rtmdet.git
	 cd rtmdet
	 ```

2. (Optional) Create and activate a Python virtual environment:
	 ```bash
	 python3 -m venv venv
	 source venv/bin/activate
	 ```

3. Install dependencies:
	 ```bash
	 pip install torch torchvision
	 # Add other dependencies as needed
	 ```

## Citation

If you use this codebase, please cite the original RTMDet paper.

```
@misc{lyu2022rtmdet,
      title={RTMDet: An Empirical Study of Designing Real-Time Object Detectors},
      author={Chengqi Lyu and Wenwei Zhang and Haian Huang and Yue Zhou and Yudong Wang and Yanyi Liu and Shilong Zhang and Kai Chen},
      year={2022},
      eprint={2212.07784},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```