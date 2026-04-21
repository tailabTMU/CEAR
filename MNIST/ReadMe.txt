MNIST Robust Training (Teacher / Distillation / Ensembles)

This repo contains a TensorFlow/Keras training script for MNIST that supports:
- Training a clean teacher CNN
- Training noisy single-network baselines (RS / “SOTA_WE”)
- Training defensive distillation students:
  - Proposed WVGA (variable Gaussian augmentation: beta varies around sigma)
  - Proposed WOVGA (without variable Gaussian augmentation: beta = sigma)
- Building ensembles from trained students:
  - Weighted Ensemble (WWE-style averaging using test accuracies)
  - Majority Voting (uniform averaging)

--------------------------------------------------

DATASET
MNIST is downloaded automatically by Keras:
- tf.keras.datasets.mnist.load_data()

No manual dataset download is required.

Preprocessing performed by the script:
- Convert to float32 and normalize to [0,1]
- Add a channel dimension (28,28) -> (28,28,1)
- One-hot encode labels to 10 classes

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
mode   : Select what to run (see map below)
modelNo: student identifier used in filenames and (for WVGA) beta selection
temp   : distillation temperature (recommended range: 2–7)

--------------------------------------------------

MODEL
Base CNN (create_base_model):
- Conv2D(32) -> Conv2D(64) -> MaxPool
- Conv2D(64) -> Conv2D(64) -> MaxPool
- Flatten -> Dense(128) -> Dense(64) -> Dense(10, softmax)

Noise:
Gaussian noise is applied in NumPy:
noisy_images = clip(images + N(0, sigma), 0, 1)

--------------------------------------------------

MODE MAP (WWE_or_base)

IMPORTANT: The help string says “1: Teacher …” but the actual __main__ mapping is:

1 : Build Weighted Ensemble (loads pre-trained WOVGA students)
2 : Train teacher (clean training)
3 : Train noisy baseline single networks (RS_WE)
4 : Train WOVGA student (defensive distillation, beta = sigma)
5 : Build Majority Voting ensemble
Else (any other integer) : Train proposed WVGA student (variable beta)

--------------------------------------------------

OUTPUTS / CHECKPOINT PATHS

Teacher:
MNIST_updated_model/MNISTTeacher.h5

Noisy baseline single networks (RS / SOTA_WE):
MNIST_updated_model/SOTA_WE/RS_WE_modelNo{temp}_noise{sigma}.h5
(Note: temp here is the loop index i+5 inside train_RS_WE.)

Proposed WVGA student:
MNIST_updated_model/mnist_defensive_distillation_T_{modelNo}_noise_{sigma}.h5

Proposed WOVGA student:
MNIST_updated_model/WithoutVGA/mnist_defensive_distillation_T_{modelNo}_noise_{sigma}.h5

Weighted Ensemble output:
MNIST_updated_model/WithoutVGA/Proposed_WOVGA_WWE_{sigma}.h5

Majority Voting output:
MNIST_updated_model/Proposed_MVE/WVGA/Proposed_WVGA_MV_{sigma}.h5

RECOMMENDED: Create directories first (to avoid save errors):
mkdir -p MNIST_updated_model
mkdir -p MNIST_updated_model/WithoutVGA
mkdir -p MNIST_updated_model/SOTA_WE
mkdir -p MNIST_updated_model/Proposed_MVE/WVGA

--------------------------------------------------

EXAMPLES

1) Train the teacher (mode=2)
python train.py 0.25 2 1 5

2) Train noisy baseline RS_WE models (mode=3)
- Trains multiple noisy models internally (looped)
python train.py 0.25 3 1 5

3) Train WOVGA defensive distillation student (mode=4)
- Requires teacher checkpoint:
  MNIST_updated_model/MNISTTeacher.h5
python train.py 0.25 4 3 5
# sigma=0.25, modelNo=3, temp=5

4) Train proposed WVGA defensive distillation student (mode=any value other than 1..5)
- Requires teacher checkpoint:
  MNIST_updated_model/MNISTTeacher.h5
python train.py 0.25 9 2 6
# mode=9 triggers train_proposed(...)

5) Build weighted ensemble (mode=1)
- Expects WOVGA student checkpoints to exist (see “Gotchas” about paths):
python train.py 0.25 1 1 5

6) Build majority voting ensemble (mode=5)
python train.py 0.25 5 1 5

--------------------------------------------------

IMPORTANT NOTES / GOTCHAS

- Train the teacher FIRST (mode=2) before running any distillation (mode=4 or “else”).
- Distillation uses teacher predictions as “soft labels”:
  soft_labels = teacher.predict(x_train)
- Proposed WVGA (train_proposed):
  - Chooses beta around sigma using:
    var = sigma/2
    beta = sigma +/- uniform(0, var) depending on temp (temp==1, temp%3==0, else).
  - Trains on noisy_x_train with beta, validates on noisy_x_test with sigma.
- Proposed WOVGA (train_without_VGA):
  - Uses beta = sigma for both train and test noise.

PATH MISMATCHES (you may need to adjust depending on where your models are saved):
- WeightedEnsemble currently loads from:
  MNIST_updated_model/WithoutVGA/mnist_WOVGA_noise_{sigma}/mnist_defensive_distillation_T_{i}_noise_{sigma}.h5
  but train_without_VGA saves to:
  MNIST_updated_model/WithoutVGA/mnist_defensive_distillation_T_{modelNo}_noise_{sigma}.h5

- MajorityVoting currently loads from:
  MNIST_updated_model/mnist_noise_{sigma}/mnist_defensive_distillation_T_{i}_noise_{sigma}.h5
  but train_proposed saves to:
  MNIST_updated_model/mnist_defensive_distillation_T_{modelNo}_noise_{sigma}.h5

If you want ensembles to work out-of-the-box, either:
1) Update the save paths to match the loader paths, OR
2) Update the loader paths to match the save paths.

--------------------------------------------------
