## Gradual sensory maturation promotes abstract representation learning

Jeonghwan Cheon*, Se-Bum Paik†

\* First author: jeonghwan518@kaist.ac.kr  
† Corresponding author: sbpaik@kaist.ac.kr  


### Associated article

This repository contains the implementation and demo codes for the manuscript "**Gradual sensory maturation promotes abstract representation learning**" (currently under submission).

#### Abstract

Human infants begin life with limited visual capacities, such as low acuity and poor color sensitivity, due to gradual sensory maturation. In contrast, machine learning models are trained on high-fidelity inputs from the outset, often leading to shortcut learning and overfitting to spurious correlations. Here, we show that early sensory immaturity plays a critical role in shaping bias-resistant, abstract visual representations that conventional models struggle to develop. Using neural network simulations and human psychophysics experiments, we demonstrate that gradual sensory development supports the emergence of robust and generalizable internal representations, reduces reliance on superficial cues, and promotes disentangled representations that enable compositional reconstruction and visual imagination. Comparative analyses of human and model behavior reveal shared patterns of bias resistance and adaptive generalization, including resilience to misleading information. Our findings suggest that gradual sensory maturation is not merely a developmental constraint, but rather a key mechanism that enables abstract representation learning.

#### Research highlights

- The essential yet overlooked role of gradual sensory maturation was explored
- Early sensory immaturity promotes abstract representations resistant to shortcut learning
- Emergent representations support compositional reconstruction of novel visual attributes
- Human and model behaviors show similar bias resistance and adaptive generalization

### Simulations and experiments

Each simulation and experimental result can be replicated by running the corresponding demo files in the ```\scripts``` directory. The detailed and core implementation of the model can be found in the ```\src``` directory.

#### Prerequisites

- Python 3.12 (Python software foundation)
- Pytorch 2.6.0
- NumPy 2.2.3
- SciPy 1.15.2

