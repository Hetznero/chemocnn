# Chemocnn

In this project we develop the connectivity of convolutional neural networks with a
chemoaffinity-based concept. Every neuron gets a ligand value and a receptor value. To decide
whether a presynaptic neuron *i* connects to a postsynaptic neuron *j*, we compare the receptor
value of *i* to the ligand value of *j*, and a connectivity rule turns the mismatch of the two
into a connection probability:

```
p(i→j) = c( d(r_i, ℓ_j) )
```

Here `d` is a distance function and `c` a connectivity rule. Each layer holds an excitatory population, an inhibitory
population, or both, and all connections run feedforward from one layer to the next. Nothing is
learned from data, the connectivity comes entirely out of the development rule.

With this concept we build a 1D On-Center Cell model, which produces the Mexican Hat receptive
fields of on-center cells. We then extend it to a 2D Visual System model that also produces
orientation-selective simple cells and phase-invariant complex cells. Finally, we build two-layer
networks that recreate a specific kernel, namely simple cell receptive fields and Gabor filters,
where we obtain the Gabor filters both from a Gaussian mixture model and by sampling the noise
from a Gabor filter directly.

We analyze the resulting cells with radial and orientation tuning curves, with receptive fields
obtained by reverse correlation, and with F1/F0 scores from drifting sinusoidal gratings.

## Structure

```
lib/          functions for the simulations
simulations/  jupyter notebooks that run the models and produce the figures
```

## Running the simulations

Set up a virtual environment and install the requirements:

```bash
python -m venv chemocnn
source chemocnn/bin/activate      
pip install -r requirements.txt
```

The simulations are Jupyter notebooks. Start Jupyter and open any notebook in `simulations/`:

```bash
jupyter lab
```