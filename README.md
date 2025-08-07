# LIRDRec
Source code for [LIRDRec](https://arxiv.org/abs/2505.04960) is located at `src/models/lirdrec.py`.

## Dependencies
The script has been tested running under Python 3.7.11, with the following packages installed (along with their dependencies):
- `torch: 1.11.0+cu113`
- `pandas: 1.3.5`
- `pyyaml: 6.0`
- `numpy: 1.21.5`
- `scipy: 1.7.3`
- `sentence-transformers: 2.2.0`

## Datasets 
- The model is based on [MMRec](https://github.com/enoche/MMRec). Please refer to the datasets used in the MMRec framework on GitHub.

## How to run
1. Put your downloaded data (e.g. `baby`) under `data` dir.
2. Enter `src` folder and run with:  
`python main.py -m lirdrec -d baby`  
You may specify other parameters in CMD or config with `configs/model/LIRDRec.yaml` and `configs/dataset/*.yaml`.
For reproducibility, model settings are specified in `configs/model/LIRDRec.yaml`.
