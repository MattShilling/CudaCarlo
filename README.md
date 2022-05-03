# CudaCarlo
Using CUDA w/ a Monte Carlo Simulation

# Building the Monte Carlo Simulation

This project was built on a very specific system: the OSU `rabbit` server. In order to run this properly on another system, you will have to replace the included `helper_*` header paths needed to compile with the location of them on your own system. Other than that? Just run `make`.

Oh and you need a NVIDA CUDA available graphics card.

# Running the Monte Carlo Simulation

Use: `./CudaCarlo [number of threads per block] [number of trials to run]`
