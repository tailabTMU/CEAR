\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}

\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single
}

\title{Project README (Training + Certification)}
\date{}

\begin{document}

\maketitle

This repo contains training scripts for three datasets (Tiny-ImageNet, CIFAR-10, MNIST) and a certification script \texttt{certify.py} for randomized smoothing. Each training script produces \texttt{.h5} model checkpoints that can later be used by \texttt{certify.py} to compute certified radii.

\section{Tiny-ImageNet Training Script (ResNet18, 64x64, 200 classes)}

\subsection{What it does}
\begin{itemize}
    \item Loads Tiny-ImageNet-200 from disk (must be downloaded manually)
    \item Trains a ResNet18-style teacher model
    \item Trains multiple student models with noise + defensive distillation
    \begin{itemize}
        \item WVGA: variable noise beta around sigma
        \item WOVGA: beta = sigma
    \end{itemize}
    \item Trains baselines (RandSmooth-style, SWEEN)
    \item Builds ensembles:
    \begin{itemize}
        \item Weighted Ensemble (accuracy-weighted averaging)
        \item Majority Voting (uniform/fixed weights)
    \end{itemize}
\end{itemize}

\subsection{Key outputs}
\begin{lstlisting}
ImagenetModels/ImageNetTeacher.h5
ImagenetModels/Proposed/WVGA/Noise_{sigma}/Student_network{sigma}_modelNo..._temp{T}.h5
ImagenetModels/Proposed/WOVGA/Noise_{sigma}/Student_network{sigma}_modelNo..._temp{T}.h5
ImagenetModels/Proposed/WOVGA/Noise_{sigma}/Proposed_WOVGA_WE_imgnet_{sigma}.h5
\end{lstlisting}

\subsection{Dataset requirement}
Download Tiny-ImageNet-200 and extract to:
\begin{lstlisting}
tiny-imagenet-200/train
tiny-imagenet-200/val
\end{lstlisting}

Download: \url{http://cs231n.stanford.edu/tiny-imagenet-200.zip}

\section{CIFAR-10 Training Script (ResNet-110, 32x32, 10 classes)}

\subsection{What it does}
\begin{itemize}
    \item Loads CIFAR-10 automatically via \texttt{tf.keras.datasets.cifar10.load\_data()}
    \item Trains a ResNet-110 teacher
    \item Trains noisy single-network baselines (SOTA / RS-style)
    \item Trains defensive distillation students:
    \begin{itemize}
        \item proposed\_WVGA: variable beta around sigma
        \item proposed\_WOVGA: beta = sigma
    \end{itemize}
    \item Builds ensembles:
    \begin{itemize}
        \item Weighted Ensemble from WOVGA students
        \item Majority Voting from WOVGA students
    \end{itemize}
\end{itemize}

\subsection{Key outputs}
\begin{lstlisting}
Cifar_updated_model/WithNoise/Cifar10Teacher.h5
Cifar_updated_model/WithNoise/SOTA/RS_WE_sigma{sigma}_modelno{modelNo}.h5
Cifar_updated_model/WithNoise/proposed_WVGA/Student_network{sigma}_temp{modelNo}.h5
Cifar_updated_model/WithNoise/proposed_WOVGA/Student_network{sigma}_temp{modelNo}.h5
Cifar_updated_model/Proposed_WOVGA_WWE_{sigma}.h5
\end{lstlisting}

\subsection{Notes}
A cleaned \texttt{train\_model()} version matches the Tiny-ImageNet distillation style:
teacher predicts probabilities $\rightarrow$ offline temperature smoothing $\rightarrow$ student trained with KD loss.

Requested save path:
\begin{lstlisting}
Cifar_updated_model/WithNoise/proposed_WVGA/Student_network{sigma}_temp{modelno}.h5
\end{lstlisting}

Temperature range: 2--7

\section{MNIST Training Script (CNN, 28x28, 10 classes)}

\subsection{What it does}
\begin{itemize}
    \item Loads MNIST via \texttt{tf.keras.datasets.mnist.load\_data()}
    \item Trains a teacher CNN (clean)
    \item Trains noisy baseline single networks
    \item Trains defensive distillation students:
    \begin{itemize}
        \item Proposed WVGA (train\_proposed)
        \item Proposed WOVGA (train\_without\_VGA)
    \end{itemize}
    \item Builds ensembles:
    \begin{itemize}
        \item Weighted Ensemble (WWE)
        \item Majority Voting
    \end{itemize}
\end{itemize}

\subsection{Key outputs}
\begin{lstlisting}
MNIST_updated_model/MNISTTeacher.h5
MNIST_updated_model/SOTA_WE/RS_WE_modelNo{temp}_noise{sigma}.h5
MNIST_updated_model/mnist_defensive_distillation_T_{modelNo}_noise_{sigma}.h5
MNIST_updated_model/WithoutVGA/mnist_defensive_distillation_T_{modelNo}_noise_{sigma}.h5
MNIST_updated_model/WithoutVGA/Proposed_WOVGA_WWE_{sigma}.h5
MNIST_updated_model/Proposed_MVE/WVGA/Proposed_WVGA_MV_{sigma}.h5
\end{lstlisting}

\subsection{Important notes}
\begin{itemize}
    \item The \texttt{\_\_main\_\_} mode map differs from argparse help text.
    \item Check the if/elif chain at the bottom of the script.
    \item Ensemble load paths may not match save paths.
\end{itemize}

\section{certify.py (Randomized Smoothing)}

\subsection{What it does}
\begin{lstlisting}
from datasets import get_dataset, DATASETS, get_num_classes
\end{lstlisting}

\begin{lstlisting}
prediction, radius = smoothed_classifier.certify(x, N0, N, alpha, batch)
\end{lstlisting}

Outputs:
\begin{lstlisting}
idx    label    predict    radius    correct    time
\end{lstlisting}

\subsection{Configurations}

\textbf{Single network (NO GM)}
\begin{lstlisting}
from core import Smooth
\end{lstlisting}

\textbf{Multi-network (Geometric Median)}
\begin{lstlisting}
from core_GM import Smooth
\end{lstlisting}

\textbf{Important:}
You must manually update \texttt{.h5} filenames in \texttt{certify.py}.

\subsection{Sample run}
\begin{lstlisting}
nohup python3 certify.py imagenet ImagenetModels/Proposed/WVGA/Noise_0.25/ 0.25 ImagnetResults/Proposed_GM_WVGA/imgnet0.25_results_cont2 > test.out &
\end{lstlisting}

\subsection{Tips}
\begin{itemize}
    \item Ensure paths end with ``/''
    \item Modify \texttt{start\_ind = 2160} if needed
    \item Use \texttt{--skip} and \texttt{--max}
\end{itemize}

\section{Quick Workflow}
\begin{enumerate}
    \item Train models
    \item Build ensemble (optional)
    \item Run \texttt{certify.py}
\end{enumerate}

\section{Environment Requirements}
Tested with TensorFlow 2.15.

\subsection{CUDA / cuDNN}
Example:
\begin{itemize}
    \item CUDA 11.8
    \item cuDNN 8.6
\end{itemize}

\subsection{Verify GPU}
\begin{lstlisting}
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
\end{lstlisting}

\end{document}