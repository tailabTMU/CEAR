certify.py (Randomized Smoothing Certification)

This folder contains the main certification script: certify.py.
It certifies examples using a (possibly ensemble) base classifier wrapped by a randomized smoothing
implementation (Smooth), and writes per-example certification results to a tab-separated output file.

--------------------------------------------------

WHAT THIS SCRIPT DOES

Given:
- A dataset (mnist / cifar10 / imagenet / etc. as supported by datasets.py)
- A directory containing saved TensorFlow/Keras models (base classifiers)
- A noise level sigma for randomized smoothing

It will:
1) Load the dataset split (train or test)
2) For a subset of examples, run:
   prediction, radius = smoothed_classifier.certify(x, N0, N, alpha, batch)
3) Write results to outfile with columns:
   idx    label    predict    radius    correct    time

--------------------------------------------------

REQUIREMENTS

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Project modules present in this repo:
  - datasets.py (or datasets/ package) providing: get_dataset, DATASETS, get_num_classes
  - core.py and/or core_GM.py providing: Smooth
  - (Optional) any dataset-specific utilities used by datasets.py

Install:
pip install tensorflow numpy

--------------------------------------------------

CLI ARGUMENTS

Positional:
dataset          : one of DATASETS (from datasets.py)
base_classifier  : path (directory) to saved TF/Keras model(s)
sigma            : noise hyperparameter for randomized smoothing
outfile          : output file path (tsv)

Optional:
--batch   : batch size for sampling in certify() (default: 1000)
--skip    : certify every k-th example (default: 20)
--max     : stop after this many indices (default: 3500)
--split   : train or test (default: test)
--N0      : number of samples for initial class selection (default: 100)
--N       : number of Monte Carlo samples for certification (default: 100000)
--alpha   : failure probability (default: 0.001)

--------------------------------------------------

IMPORTANT: THIS IS THE MAIN FOLDER / ENTRYPOINT

This certify.py folder is the main folder you run from (it is the entrypoint for certification).
Run commands from the directory that contains certify.py so imports like:
- from datasets import get_dataset, DATASETS, get_num_classes
- from core import Smooth   OR   from core_GM import Smooth
resolve correctly.

--------------------------------------------------

TWO MODES OF OPERATION

The code supports two different “base classifier” configurations:

(1) Single Network (NO Geometric Median / NO GM)
(2) Multi-Network Voting with Geometric Median (GM)

You select which one runs by commenting/uncommenting the blocks inside certify.py.

--------------------------------------------------

(1) SINGLE NETWORK (NO GM)

To run all models WITHOUT GM:
- Configure certify.py such that the “For Single Network” block is active
- Comment out the “For Mul Network” block
- Import Smooth from core (non-GM):

Use this import:
from core import Smooth  # core_og = no GM

And activate the following block:

    # ------------- For Single Network ------------------------------------ #
    model = tf.keras.models.load_model(args.base_classifier + "RandomizedSmoothing_single_Cifar10_noise_0.5.h5", compile=False)
    model = tf.keras.models.load_model(args.base_classifier + "Proposed_WOVGA_WWE_1.0.h5", compile=False)
    model.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
    arr_models.append(model)

Notes:
- This block appends ONE compiled model to arr_models, which is then used as the base classifier.
- Make sure the filename you want is the one you keep (the snippet shows two load_model lines;
  typically you would keep ONE of them, or ensure the second is intended to overwrite the first).

--------------------------------------------------

(2) MULTI NETWORK + GEOMETRIC MEDIAN (GM)

For voting with geometric median:
- Keep the “For Mul Network” block active
- Comment out the “For single Network” block
- Import Smooth from core_GM:

Use this import:
from core_GM import Smooth  # core_GM = for GM

And activate the following block:

    #------------ For Mul Network ------------ #
    for i in range(1,6):
        model = tf.keras.models.load_model(args.base_classifier + "Student_network"+str(args.sigma)+"_modelNo"+str(i)+".h5", compile=False)
        # model = tf.saved_model.load(file_model)
        # #model = datamodel(file_model, sess)
        arr_models.append(model)

    base_classifier = arr_models

Notes:
- This loads 5 student networks and passes them into Smooth.
- The expectation is that core_GM.Smooth aggregates the ensemble predictions using a GM-based rule.

--------------------------------------------------

OUTPUT FORMAT

The output file is tab-separated with header:
idx    label    predict    radius    correct    time

Where:
- idx     : dataset index
- label   : true label (decoded from one-hot for mnist/cifar10/imagenet)
- predict : predicted class from smoothed classifier
- radius  : certified radius
- correct : 1 if predicted == true label else 0
- time    : elapsed time for certification of that sample

--------------------------------------------------

SAMPLE COMMAND

Example (as provided):

nohup python3 certify.py imagenet ImagenetModels/Proposed/WVGA/Noise_0.25/ 0.25 ImagnetResults/Proposed_GM_WVGA/imgnet0.25_results_cont2 > test.out &

Meaning:
- dataset = imagenet
- base_classifier directory = ImagenetModels/Proposed/WVGA/Noise_0.25/
- sigma = 0.25
- outfile = ImagnetResults/Proposed_GM_WVGA/imgnet0.25_results_cont2
- logs redirected to test.out, running in background via nohup

--------------------------------------------------

TIPS / GOTCHAS

- Ensure base_classifier path ends with a trailing “/” because the script concatenates strings:
  args.base_classifier + "Student_network..."
- start_ind is currently hardcoded (start_ind = 2160). Change it if you want to start earlier.
- If you want to certify ALL examples, set:
  --skip 1
  and increase --max (or remove that stopping condition in code).
- If your labels are not one-hot for a dataset, check datasets.py formatting. The script has
  separate branches for mnist/cifar10/imagenet vs other datasets.
