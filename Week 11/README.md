# Session 11 - Super Convergence

## Results
 * Network as described is implemented (DevidNET)
 * OneCycleLR sheduler is used:
   - min_lr = 0.1
   - max_lr = min_lr/5
   - Step up till epochs 5 and and stepdown till the end of epochs 24
   - Epochs = 24
 * Achived expected validation accuracy of **90%**
 * Cylic LR
 * Batch size = 512
 * transform used: Padding-->Randomcrop-->flip-->cutout

## Accuracy and Loss
![i](images/acc.png) ![i2](images/loss.png)

## LRfinder with range test
![img3](images/lrf.png)

## Cyclic LR plot
![img5](images/download.png)
