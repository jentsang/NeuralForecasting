# HDR ML Challenge year 2: Scientific Modeling out of distribution (Scientific-Mood)

## Problem setting: Neural Forecasting
To forecast the activations of a cluster of neurons given previous signals from the same cluster. This targets the critical problem of brain-artificial neuron interfaces, and these models can be used in brain-chip interfaces for artificial limb control, amongst many others.

## Datasets
The motor neural activity forecasting dataset includes recorded neural signals from two monkeys performing reaching activities, Monkey A and Monkey B, using μECoG arrays. Recorded neural signals are in the form of multivariate continuous time series, with variables corresponding to recording electrodes in the recording array.

The dataset includes all 239 electrodes from Monkey A and 87 electrodes specifically from the M1 region of Monkey B.

More information about the dataset can be found [here](https://www.codabench.org/competitions/9806/#/pages-tab).

## Repository Structure

```text
├── literature/         # Relevant papers and background documentation
├── notebooks/          # Jupyter notebooks for data exploration, EDA, and model training
├── src/                # Source code and helper functions
├── starting_kit/       # Initial challenge materials and raw data setup
├── submission/         # Final model scripts and weights prepared for Codabench
├── .gitignore          # Git ignore rules
├── LICENSE             # MIT License
└── README.md           # Project documentation
```

## Methodology & Experimental Learnings
Achieving a highly competitive benchmark required an iterative, empirical approach to feature selection, model architecture, and training dynamics tailored specifically to biological time-series data. 

Through rigorous hypothesis testing, several key insights drove the final model design:

### 1. Feature Selection: Signal Over Noise
Initial baselines utilized 9 different frequency bands. However, empirical testing revealed that feeding the model all bands acted as distracting noise. Isolating and training exclusively on the **raw LFP signal (Feature 0)** dramatically improved the $R^2$ score by allowing the network to focus on the core continuous wave. 

### 2. Architecture Reality Checks
Several state-of-the-art architectures were stress-tested against the continuous, causal nature of brain waves:
* **Transformers (Multi-Head Attention):** Failed to generalize (yielding negative $R^2$ scores). Without a strict sequential bias, the model struggled to understand the forward-moving momentum of continuous physiological waves over a short 10-step window.
* **1D-CNN Hybrids:** Convolutional feature extractors "blurred" the precise, step-by-step timing required for this specific forecasting task, degrading performance.
* **Bidirectional GRUs:** Reading the 10-step history backward scrambled the natural physical causality of the signal, proving that for this dataset, time must strictly flow in one direction.
* **The Champion Architecture:** The undisputed best performer was a **Unidirectional, 3-Layer GRU with a 2048-width hidden size**. This deep, focused architecture possessed the sequential bias needed to respect causality and the raw capacity to memorize complex wave shapes.

### 3. Training Dynamics & Stabilization
Training deep recurrent networks on μECoG arrays required specialized regularization techniques to prevent gradient explosion and late-stage overfitting:
* **Loss Function Strategy (Huber Loss):** While the competition is evaluated on Mean Squared Error (MSE), training the model using `SmoothL1Loss` (Huber Loss) acted as a shock absorber against random biological spikes and artifacts. This resulted in exceptionally stable training curves without violent weight updates.
* **Gradient Clipping:** Essential for preventing the GRU from tearing itself apart during Backpropagation Through Time (BPTT). 
* **Regularization:** Heavy Dropout (0.3) and Weight Decay (L2 regularization) were utilized to prevent the model from overfitting to the training data by Epoch 100, forcing it to learn generalized wave patterns instead.