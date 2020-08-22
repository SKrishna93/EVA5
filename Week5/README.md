# Session 5

## Model 1

### Objective
* Build a base network for our model. Inspired from previous assignment
* Target Test and Train accuracy of above 98%
* < 20k parameters and < 20 Epochs

### Summary
* Basic model with 19k parameters
* Test Accuracy: 99.29%
* Train Accuracy: 98.89%
* Slight overfitting observed in the model (Can be fixed in the subsequent iterations by adding BatchNorm and Regulariaztion)
* Network is built with two convolution blocks each followed by two transition blocks
* First transistion block is added after 3 layers of convolution
* Second transistion block is added after 2 more layers of convolution

## Model 2

### Objective

* Reduce the number of parameters
* Achieve 99% Test accuracy with <15 Epochs

### Summary
* Model is working as expected
* Parameters: 13k
* Train Accuracy: 99.06%
* Test Accuracy: 99.01%
* No over fitting observed in the model
* Model can be pushed further to reduce the number to parameters and improve accuracy
* Add BatchNorm to help the model learn faster (Lower the number of epochs)

# Model 3

### Objective

* <10k parameters
* Help model learn faster by using BatchNorm

### Summary
* Reduced the model parameters to 9.6k
* Train Accuracy: 99.67%
* Test Accuracy: 99.43%
* Used BatchNorm, model is learning faster. Achieving 98% accuracy in epoch 2
* Model is grossly overfitting. Training accuracy is consistently increasing while Test accuracy is struggling to keep up.

Use regularization techiniques like dropout. Image Augmentation to make model sweat harder during training :)    

# Model 4

### Objective

* Add Image Augmentation (Random Rotation) to force model to train harder
* Achieve 99.4% or higher accuracy consistently 

### Summary
* Train Accuracy: 99.17%
* Test Accuracy: 99.44%
* Overfitting issue has been taken care of. Test acuracy is higher than train accuracy in all epochs
* Getting consistently >99.4% accuracy in last few epochs

# Model 5 (Final Model)

### Objective

* Add step LR scheduler decay the learning rate by 0.1 after every 6 steps 

### Summary
* Train Accuracy: 99.14%
* Test Accuracy: 99.55%
* Significant improvement in the Test accuracy of the model after adding LR scheduler
* Model is able to cross the 99.4% test accuracy threshold faster and able to maintain it throughout the remaining epochs
  
# Model 6 (Mission Impossible)
### Objective

* Reduce the number of parameters in the Model to <8K

### Summary
* Parameter=5.6K
* Train Accuracy: 98.70%
* Test Accuracy: 99.50%
* Drop in Train Accuracy just as expected, as a result of Low number of channels extracted in the model
* Model is able to cross the 99.4% test accuracy
* A good scheduler does wonders! :)
