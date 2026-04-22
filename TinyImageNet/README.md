
# Tiny-ImageNet Robust Training (Teacher / Students / Ensembles)

This repo contains a TensorFlow/Keras training script for Tiny-ImageNet-200 that supports:
- Training a teacher model (ResNet18-style)
- Training noisy baselines and SWEEN variants
- Training defensive distillation students (with and without variable Gaussian augmentation)
- Building ensembles (weighted ensemble + majority voting style averaging)

### DATASET (REQUIRED)
You must download Tiny-ImageNet-200:

**Download:**
`http://cs231n.stanford.edu/tiny-imagenet-200.zip`

Setup:
`wget http://cs231n.stanford.edu/tiny-imagenet-200.zip`
`unzip tiny-imagenet-200.zip`

The dataset must be located at:
`tiny-imagenet-200/train`
`tiny-imagenet-200/val`

If stored elsewhere, edit these lines in the script:
`image_dataset_from_directory("tiny-imagenet-200/train")`
`image_dataset_from_directory("tiny-imagenet-200/val")`

---

## ARGUMENTS
`python train.py <sigma> <mode> <modelNo> <temp>`

- `sigma`  : Gaussian noise std-dev $\sigma$ (example: 0.25)
- `mode`  : Select training/evaluation mode
- `modelNo`: student model identifier
- `temp`   : Distillation temperature (VALID RANGE: 2–7)

---

MODE MAP
- **1** : Train teacher
- **2** : Train RandSmooth baseline
- **3** : Train SWEEN
- **4** : Weighted Ensemble
- **5** : Proposed WVGA distillation
- **6** : Evaluate/debug
- **7** : Proposed WOVGA distillation
- **8** : Majority voting ensemble

--- 

## EXAMPLES

### Train teacher:
- `python train.py 0.25 1 1 2`

### Train RandSmooth baseline (sigma=0.25):
- `python train.py 0.25 2 1 2`

### Train SWEEN models:
- `python train.py 0.25 3 1 2`

### Train proposed WOVGA students (requires teacher):
- `python train.py 0.25 7 1 3`

### Train proposed WVGA students (requires teacher):
- `python train.py 0.25 5 1 3`

### Create weighted ensemble:
- `python train.py 0.25 4 1 2`

### Create majority voting ensemble:
- `python train.py 0.25 8 1 2`

---

## ⚠️ Important Notes
- Train the teacher **first** before distillation modes
- Requires large RAM (datasets converted to NumPy)
- Ensure checkpoint directories exist:

    ImagenetModels/
    ImagenetModels/RandSmoth/
    ImagenetModels/Sween/
    ImagenetModels/Proposed/WVGA/
    ImagenetModels/Proposed/WOVGA/
