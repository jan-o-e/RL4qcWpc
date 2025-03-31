# Create an environment and give it an appropriate name
export CONPREFIX=qiskit
conda create --prefix $CONPREFIX

# Activate your environment
conda activate $CONPREFIX

# Install packages...
#conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
#pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
conda install cuda -c nvidia
pip install --upgrade "jax[cuda12]"
pip install qiskit-dynamics
pip install gymnax evosax distrax optax flax numpy brax wandb flashbax diffrax