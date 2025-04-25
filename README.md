# EdgeCodec
EdgeCodec: Onboard Lightweight High Fidelity Neural Compressor with Vector Quantization

## Introduction

On this git you will find the essentials to _EdgeCodecs_ structure, training and evaluation. _EdgeCodec_ relies on [lucidrains vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) repository for the residual vector quantizer implementation.

<div style="background-color: white; padding: 10px; display: inline-block;">
  <img src="figures/full_detail_merged_model.png" alt="Example Image" width="700">
</div>


## Dataset

This git does NOT provide the original _AeroSense_ dataset, which _EdgeCodec_ is trained on. For access to the dataset and it's associated processing library _Aerolib_ please contact xxx. 

If you want to test the code as it is, you will need to provide tensors of shape _batch_size_ x 36 x 800. 

## Requirements

