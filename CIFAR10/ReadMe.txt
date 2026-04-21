CIFAR-10 Robust Training (Teacher / Students / Ensembles)

This repo contains a TensorFlow/Keras training script for CIFAR-10 that supports:
- Training a teacher network (ResNet-110)
- Training single noisy networks (SOTA/RS baseline)
- Training defensive distillation (DD) student networks:
  - Proposed WVGA (variable Gaussian augmentation: beta varies around sigma)
  - Proposed WOVGA (without variable Gaussian augmentation: beta = sigma)
- Building ensembles from trained students:
  - Weighted Ensemble (WWE-style averaging using validation accuracies)
  - Majority Voting (uniform averaging)

--------------------------------------------------

DATASET
CIFAR-10 is downloaded automatically by Keras:
- tf.keras.datasets.cifar10.load_data()

No manual dataset download is required.

The script normalizes images to [0, 1] via:
x_train, x_test = x_train / 255.0, x_test / 255.0

--------------------------------------------------

REQUIREMENTS
- Python 3.8+
- TensorFlow 2.x
- NumPy

Install:
pip install tensorflow numpy

--------------------------------------------------

ARGUMENTS
python train.py <sigma> <mode> <modelNo> <temp>

sigma  : Gaussian noise standard deviation (example: 0.25)
mode   : Select training/evaluation mode (see map below)
modelNo: model identifier used to choose beta variation and save names
temp   : distillation temperature (RECOMMENDED RANGE: 2–7)

--------------------------------------------------

MODEL
Architecture: ResNet-110 (custom implementation)
- Final layer outputs logits (Dense(num_classes, activation=None))

Data Augmentation:
Uses ImageDataGenerator:
- width_shift_range=0.1
- height_shift_range=0.1
- horizontal_flip=True

Noise:
Gaussian noise is applied in NumPy:
noisy_images = clip(images + N(0, sigma), 0, 1)

--------------------------------------------------

MODE MAP (WWE_or_base)
1 : Train teacher (clean training, no noise)
2 : Train proposed WVGA student (defensive distillation, variable beta)
3 : Train noisy single network baseline (RS_WE_sigma...)
4 : Train proposed WOVGA student (defensive distillation, beta = sigma)
5 : Build weighted ensemble from proposed WOVGA students
6 : Build majority voting ensemble from proposed WOVGA students

--------------------------------------------------

OUTPUTS / CHECKPOINT PATHS

Teacher:
Cifar_updated_model/WithNoise/Cifar10Teacher.h5

Noisy baseline single networks:
Cifar_updated_model/WithNoise/SOTA/RS_WE_sigma{sigma}_modelno{modelNO}.h5

Proposed WVGA students:
Cifar_updated_model/WithNoise/proposed_WVGA/Student_network{sigma}_temp{modelno}.h5

Proposed WOVGA students:
Cifar_updated_model/WithNoise/proposed_WOVGA/Student_network{sigma}_temp{modelno}.h5

Weighted Ensemble (from proposed WOVGA students):
Cifar_updated_model/Proposed_WOVGA_WWE_{sigma}.h5

Majority Voting Ensemble (from proposed WOVGA students):
Cifar_updated_model/WithNoise/Proposed_MVE_WOVGA/Proposed_MVE_WOVGA{sigma}.h5

RECOMMENDED: Create directories first (to avoid save errors):
mkdir -p Cifar_updated_model/WithNoise
mkdir -p Cifar_updated_model/WithNoise/SOTA
mkdir -p Cifar_updated_model/WithNoise/proposed_WVGA
mkdir -p Cifar_updated_model/WithNoise/proposed_WOVGA
mkdir -p Cifar_updated_model/WithNoise/Proposed_MVE_WOVGA

--------------------------------------------------

EXAMPLES

1) Train the teacher (mode=1)
python train.py 0.25 1 1 2

2) Train a proposed WVGA student (mode=2)
- Requires the teacher checkpoint:
  Cifar_updated_model/WithNoise/Cifar10Teacher.h5
python train.py 0.25 2 3 5
# sigma=0.25, modelNo=3 (affects beta), temp=5

3) Train a noisy baseline single network (mode=3)
python train.py 0.25 3 1 2
# modelNo is used in the saved filename

4) Train a proposed WOVGA student (mode=4)
- Requires the teacher checkpoint:
  Cifar_updated_model/WithNoise/Cifar10Teacher.h5
python train.py 0.25 4 2 4

5) Build a weighted ensemble (mode=5)
- Expects proposed WOVGA student models to exist:
  Cifar_updated_model/WithNoise/proposed_WOVGA/Student_network{sigma}_temp1.h5
  ...
  Cifar_updated_model/WithNoise/proposed_WOVGA/Student_network{sigma}_temp5.h5
python train.py 0.25 5 1 2

6) Build a majority voting ensemble (mode=6)
- Same student model requirement as mode=5
python train.py 0.25 6 1 2

--------------------------------------------------

IMPORTANT NOTES / GOTCHAS

- Train the teacher FIRST (mode=1) before running distillation modes (2 or 6).
- The ResNet-110 here outputs logits (activation=None). In train_teacher() the model is
  compiled with categorical_crossentropy; this usually expects probabilities. If you see unstable
  training, change the loss to:
  tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  for teacher and/or student baselines.
- For WVGA (mode=2), the student noise beta is chosen as sigma +/- random_uniform(0, sigma/3)
  depending on modelNo (see code). For WOVGA (mode=4), beta is exactly sigma.
- Ensemble modes (5 and 6) currently load 5 student models with temp indices 1..5 and will fail
  if those checkpoints do not exist for the requested sigma.
