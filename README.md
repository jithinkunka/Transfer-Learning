# Transfer-Learning
Transfer Learning  allows us to train the neural networks with less data.


Transfer learning allows us to train deep networks using significantly less data then we would need if we had to train from scratch.
With transfer learning, we are in effect transferring the “knowledge” that a model has learned from a previous task, to our 
current one. The idea is that the two tasks are not totally disjoint, and as such we can leverage whatever network parameters that 
model has learned through its extensive training, without having to do that training ourselves.

Here we are implemetning the transfer learning using 102 flowers data set. Folders are created for every label and their respective 
flowers are moved into them. Keras is used for transfer learning here and model used is ResNet50.

