Project README (Training + Certification)

This repo contains training scripts for three datasets (Tiny-ImageNet, CIFAR-10, MNIST) and a certification script
(certify.py) for randomized smoothing. Each training script produces .h5 model checkpoints that can later be used
by certify.py to compute certified radii.

--------------------------------------------------

1) Tiny-ImageNet Training Script (ResNet18, 64x64, 200 classes)

What it does:
- Loads Tiny-ImageNet-200 from disk (must be downloaded manually)
- Trains a ResNet18-style teacher model
- Trains multiple student models with noise + defensive distillation
  - WVGA: variable noise beta around sigma
  - WOVGA: beta = sigma
- Trains baselines (RandSmooth-style, SWEEN)
- Builds ensembles:
  - Weighted Ensemble (accuracy-weighted averaging)
  - Majority Voting (uniform/fixed weights)

Key outputs (examples):
- ImagenetModels/ImageNetTeacher.h5
- ImagenetModels/Proposed/WVGA/Noise_{sigma}/Student_network{sigma}_modelNo..._temp{T}.h5
- ImagenetModels/Proposed/WOVGA/Noise_{sigma}/Student_network{sigma}_modelNo..._temp{T}.h5
- ImagenetModels/Proposed/WOVGA/Noise_{sigma}/Proposed_WOVGA_WE_imgnet_{sigma}.h5

Dataset requirement:
- Download Tiny-ImageNet-200 and extract to:
  tiny-imagenet-200/train
  tiny-imagenet-200/val
Download:
http://cs231n.stanford.edu/tiny-imagenet-200.zip

--------------------------------------------------

2) CIFAR-10 Training Script (ResNet-110, 32x32, 10 classes)

What it does:
- Loads CIFAR-10 automatically via tf.keras.datasets.cifar10.load_data()
- Trains a ResNet-110 teacher
- Trains noisy single-network baselines (SOTA / RS-style)
- Trains defensive distillation students:
  - proposed_WVGA: variable beta around sigma
  - proposed_WOVGA: beta = sigma
- Builds ensembles:
  - Weighted Ensemble from WOVGA students
  - Majority Voting from WOVGA students

Key outputs (examples):
- Cifar_updated_model/WithNoise/Cifar10Teacher.h5
- Cifar_updated_model/WithNoise/SOTA/RS_WE_sigma{sigma}_modelno{modelNo}.h5
- Cifar_updated_model/WithNoise/proposed_WVGA/Student_network{sigma}_temp{modelNo}.h5
- Cifar_updated_model/WithNoise/proposed_WOVGA/Student_network{sigma}_temp{modelNo}.h5
- Cifar_updated_model/Proposed_WOVGA_WWE_{sigma}.h5

Note on “fixed train_model” function:
- A cleaned CIFAR-10 train_model() version was produced that matches the Tiny-ImageNet distillation style:
  teacher predicts probabilities -> offline temperature smoothing -> student trained with KD loss.
- The requested save path for the fixed WVGA student is:
  Cifar_updated_model/WithNoise/proposed_WVGA/Student_network{sigma}_temp{modelno}.h5

Temperature for distillation:
- Recommended/used range: 2–7

--------------------------------------------------

3) MNIST Training Script (CNN, 28x28, 10 classes)

What it does:
- Loads MNIST automatically via tf.keras.datasets.mnist.load_data()
- Trains a teacher CNN (clean)
- Trains noisy baseline single networks (RS / SOTA_WE)
- Trains defensive distillation students:
  - Proposed WVGA (train_proposed): variable beta around sigma
  - Proposed WOVGA (train_without_VGA): beta = sigma
- Builds ensembles:
  - Weighted Ensemble (WWE)
  - Majority Voting

Key outputs (examples):
- MNIST_updated_model/MNISTTeacher.h5
- MNIST_updated_model/SOTA_WE/RS_WE_modelNo{temp}_noise{sigma}.h5
- MNIST_updated_model/mnist_defensive_distillation_T_{modelNo}_noise_{sigma}.h5
- MNIST_updated_model/WithoutVGA/mnist_defensive_distillation_T_{modelNo}_noise_{sigma}.h5
- MNIST_updated_model/WithoutVGA/Proposed_WOVGA_WWE_{sigma}.h5
- MNIST_updated_model/Proposed_MVE/WVGA/Proposed_WVGA_MV_{sigma}.h5

Important note:
- The MNIST script’s __main__ “mode map” differs from its argparse help text.
  Check the if/elif chain at the bottom of the script to see what each integer runs.
- Some ensemble load paths may not match the save paths (you may need to align them).

--------------------------------------------------

4) certify.py (Main Certification Script for Randomized Smoothing)

What it does:
- Loads a dataset split using your project’s datasets module:
  from datasets import get_dataset, DATASETS, get_num_classes
- Wraps a base classifier (single model OR an ensemble) with Smooth
- Runs randomized smoothing certification per sample:
  prediction, radius = smoothed_classifier.certify(x, N0, N, alpha, batch)
- Writes tab-separated results to an output file:
  idx    label    predict    radius    correct    time

Two certification configurations:
A) Single network (NO GM)
- Activate “For Single Network” block
- Comment out “For Mul Network” block
- Import:
  from core import Smooth  # core_og = no GM

B) Multi-network voting with Geometric Median (GM)
- Activate “For Mul Network” block (loads 5 students)
- Comment out “For single network” block
- Import:
  from core_GM import Smooth  # core_GM = for GM

IMPORTANT: Update model filenames in certify.py
- You MUST manually edit the .h5 filename strings inside certify.py so they match your saved checkpoints.
  The script does not auto-discover checkpoints; wrong names will cause load_model errors.

Sample run (provided):
nohup python3 certify.py imagenet ImagenetModels/Proposed/WVGA/Noise_0.25/ 0.25 ImagnetResults/Proposed_GM_WVGA/imgnet0.25_results_cont2 > test.out &

Tips:
- Ensure base_classifier directory path ends with “/” because filenames are concatenated.
- start_ind is hardcoded (start_ind = 2160). Change it if you want to start earlier.
- Use --skip and --max to control how many examples are certified.

--------------------------------------------------

Quick Workflow (Typical)

1) Train teacher + students on a dataset
   - Tiny-ImageNet: produces ImagenetModels/... .h5 files
   - CIFAR-10: produces Cifar_updated_model/... .h5 files
   - MNIST: produces MNIST_updated_model/... .h5 files

2) (Optional) Build an ensemble (WeightedEnsemble / MajorityVoting)

3) Run certify.py using the desired model folder and sigma, and write results to an output file
   - Choose core vs core_GM depending on whether you want GM voting
   - Update .h5 filenames in certify.py to match your saved checkpoints


--------------------------------------------------
ENVIRONMENT REQUIREMENTS

This project was tested with:
- TensorFlow 2.15

You must install a CUDA and cuDNN version compatible with TensorFlow 2.15.
(Example: CUDA 11.8 + cuDNN 8.6 are commonly used with TF 2.15.)

If the versions do not match, TensorFlow may:
- Not detect the GPU
- Crash during training
- Fail when loading models

Always verify GPU availability after install:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
