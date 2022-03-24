##Model1: C1=40, epoch=15
Test Accuracy: 82.65%

# Flip the training data:
##Model2: Flip the training data, C1=40, epoch=15
Test Accuracy: 86.82%
##Model12: Flip the training data, RandomCrop(32, padding=4), C1=40, epoch=15
Test Accuracy: 88.13%
##Model19: Flip the training data, Merging original and transformed data, RandomCrop(32, padding=4), C1=40, epoch=15
Test Accuracy: 90.38%


# Increase C1:
##Model3: Flip the training data, RandomCrop(32, padding=4), C1=42, epoch=15
Test Accuracy: 87.27%
 
# Increase epoch:
##Model17: C1=40, epoch=40
Test Accuracy: 83.58%
##Model4: Flip the training data, RandomCrop(32, padding=4), C1=42, epoch=40
Test Accuracy: 91.5%
##Model8: Flip the training data, RandomCrop(32, padding=4), C1=42, epoch=70
Test Accuracy: 90.9%
##Model18: Flip the training data, RandomCrop(32, padding=4), C1=40, epoch=40
Test Accuracy:  90.74%
##Model19: Flip the training data, Merging original and transformed data, RandomCrop(32, padding=4), C1=40, epoch=40
Test Accuracy: 91.31%


# Change optimizer:
##Model5: C1=40, epoch=15, RMSprop optimizer
Test Accuracy: 82.07%
##Model15: C1=40, epoch=15, SGD(lr=0.001,momentum=0.99)
Test Accuracy:81.89%
##Model16: C1=40, epoch=15, SGD(lr=0.001,momentum=0.9)
Test Accuracy:78.54%

# Add weight decay:
##Model6: C1=40, epoch=15, weight_decay=0.001
Test Accuracy: 81.43%
##Model7: Flip the training data, RandomCrop(32, padding=4),  C1=42, epoch=40, weight_decay=0.001
Test Accuracy: 84.26%

# Dropout:
##Model9: C1=40, epoch=15, change batchnorm to dropout
Test Accuracy: 79.02%
##Model10: C1=40, epoch=15, add dropout before batchnorm
Test Accuracy: 84.8%
##Model11: Flip the training data, RandomCrop(32, padding=4), C1=42, epoch=40, add dropout before batchnorm
Test Accuracy: 84.36%
##Model13: Flip the training data, C1=42, epoch=40, add dropout before batchnorm
Test Accuracy: 81.33%
##Model14: C1=42, epoch=40, add dropout before batchnorm
Test Accuracy: 81.13%

