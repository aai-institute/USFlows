# Notes on Improving the Conditioner

## Ideas to increase the receptive field 

Depending on the approach, the increase of the receptive field might reduce the
effect of fine-grained details. E.g. when increasing the stride or adding more
pooling layers, fine-grained details can be lost due to downsampling. 

- [ ] Deeper Network
- [ ] Larger kernels - more params 
- [ ] Strides greater than 1 - downsampling 
- [ ] Dilated Convolutions - increasing the receptive field w/o downsampling 
- [ ] ...