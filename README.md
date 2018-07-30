# denNet

This a Unet application, which can used in segmentation project with multi class, I have added some loss fuction to it including
focal loss, dice loss, iou loss and more, what's more, I also add 'Squeeze-and-Excitation Networks' module to it, just do what you
want to do.

Considering the poor result in segmenting hand from nature 3 channels picture, I think the decoder
should use a more powerful structure, like VGG, ResNet and so on, because the location is really
good, but the classification is not enough for the project.
