# ResNetBuiltSVHN
ResNet model to classify SVHN
### Background

SVHN (Street View House Number) Dataset is derived from Google Street View house numbers, each image cntains a set of Arabic numbers from '0‘ to ’9'. The train set contains 73257 numbers and the test set contains 26032 numbers. The image size is 32*32 and has 3 RGB channels.

```python
train_x shape:(73257, 32, 32, 3), train_y shape:(73257,)
test_x shape:(26032, 32, 32, 3), test_y shape:(26032,)
```

### Goal

Build a model that can learn the hidden information in the training set so that the test set can be recognized with a 95% or higher accuracy.

### Baseline

Top 20 in the global public ranking achieved 95.1% correct.

### Challenges

- Image Blurred
- Image contains multiple numbers

### Procedure

1. **Data Enhancement**
    1. The label of each image is based on the center number, thus random crop to 28X28 pixels
    2. The image has rotated, thus random rotation to $\pm30$ degree
2. **Model Construction**
    1. Choose ResNet18 and VGG16 based on the size of images and size of data set
    2. Pre-training
        1. Run 5 times, each 3 epochs on ResNet18 and VGG16
        2. **ResNet:** 
            1. The accuracy of training ranged from about 40% at the start to about 88% at the end.
            2. The variance of training accuracy was low.
            3. The test accuracy ranged from about 70% to about 88%. All of the test accuracy statistics went from high to low, which was a sign of overfitting. The ability to generalize was unstable and seemed to stop too early. This could be because it was leveling off or because the set loss threshold was too high.
        3. **VGG16:**
            1. The accuracy of training started out at about 20% and then stayed between 68 and 77%.
            2. The test accuracy ranged from 80% to about 86%.
            3. VGG stopping early only stops once.
    3. Choose ResNet18 based on the potential of models, generalization performance of models.
3. **Parameter Tune**
    1. As ResNet18 training goes on, the training set loss keeps going down and the test set loss keeps jumping across.
        
        **Solution**：Resolve model instability;  solve model over-fitting
        
    2. Unstable
        
        **Possible Reasons:**
        
        - The data is random
        - The training set is picked and cropped at random, but the test set doesn't have the same ways to improve it.
        - The rate of learning was not right.
        - Low model confidence.
        
        **Solution:** Tuned from the original learning rate of 0.001 to 0.0001 over 30 iterations.
        
    3. Overfitting
        
        **Solution:** Optimize Regularization; DropOut; Batch Size Lower ; Data Enhancement.
        
    4. Retraining
        1. First, use a learning rate of 0.001 and a batch size of 256 to learn 15 epochs. 
        2. Then, use a learning rate of 0.00001 and a batch size of 128 to learn more epochs

### Result

95.608%
