## Residual Vector Quantization in C

As mentioned in the paper related to this git, I have developed a C compatible RVQ algorithm as proposed by [SoundStream](https://arxiv.org/abs/2107.03312). As a base I utilize @MiirHo3eIN private VQ repo, which you can access by contacting him via amirhmoallemi@gmail.com . Directly, I only utilize his function **cdist**, which can be found [here](./vanilla/vq_block_kernels.c). 

Provided are 1. a simple **rvq.c**, which is RVQ implemented for a single processor and 2. a fully parallelized **Frvq.c**, which can run on a custom amount of threads. I've also provided a latent space vector from my model and my 4 fully trained codebooks in float16 so you can test it out. ***Important***: These are base C implementations and **NOT** MCU specific implementations. I've chosen to publish due to the discontinuing of the GAP9 (the original MCU this was developed for). If you still wish to see the MCU specific code, just contact me and I can release it. I can also share the sdk.config if desired for GVSOC.

## how to Run

On a regular system you can just follow these steps:

``` bash
cd embedded-rvq/vanilla
mkdir build && cd build
cmake ..
make
./Frvq # or ./rvq depending on what you built
```