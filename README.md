# ProtoSAM

ProtoSAM is a medical vessel segmentation project for retinal images, centered on prototype-guided self-prompt learning and several baseline comparisons on few-shot OCTA settings.

This repository currently contains:

- Main ProtoSAM model and training scripts 
- Dataset split JSON files and split generation scripts


## Data Preparation

Place datasets under dataset/ with the following structure.

### CHASEDB1 (for stage-1 CFP pretraining)

```text
dataset/CHASEDB1/
├── images/
└── 1stho/
```

The split file is expected at json/CHASEDB1_split.json.

### OCTA-3M / OCTA-6M

```text
dataset/OCTA-3M/
├── img/
└── gt/

dataset/OCTA-6M/
├── img/
└── gt/
```

### ROSE-1

```text
dataset/ROSE-1/
├── train/
│   ├── img/
│   └── gt/
└── test/
    ├── img/
    └── gt/
```

## Generate/Refresh Split JSON

You can regenerate split files using:

```bash
python json/generate_octa_splits.py --dataset_root dataset --output_dir json
```

Current script defaults are mainly enabled for OCTA-3M (other datasets are kept in comments in that script).

## Training

### Stage-1: CHASEDB1 pretraining

```bash
python models/train_cfp.py \
  --cfp_root dataset/CHASEDB1 \
  --cfp_json_path json/CHASEDB1_split.json \
  --output_dir output_cfp_stage1
```

Main output:

- Best checkpoint in output_cfp_stage1/.../best.pth
- TensorBoard logs in output_cfp_stage1/.../logs

### Stage-2: OCTA few-shot training

```bash
python models/train_octa.py \
  --octa_name ROSE-1 \
  --octa_root dataset/ROSE-1 \
  --octa_json_path json/ROSE-1_split.json \
  --k_shot 5 \
  --pretrained output_cfp_stage1/<run_name>/best.pth \
  --output_dir output_octa_stage2_new
```

## Checkpoint Evaluation and Visualization

Evaluate one checkpoint and save summary + overlays:

```bash
python scripts/eval_compare_checkpoints.py \
  --model sam_octa \
  --checkpoint output_baseline_compare/sam_octa/ROSE-1/k5/<run_id>/best.pth \
  --dataset ROSE-1 \
  --split test \
  --k 5 \
  --sam_vit_b_ckpt weight/sam_vit_b_01ec64.pth \
  --output_dir res_compare_eval/k=5
```

## Acknowledgements

This repository includes or references multiple upstream baseline implementations. Please check each subproject's original README and LICENSE for detailed attribution and usage constraints.
