# Detecting Masqueraders commands via Deep Learning(Ensemble)

## Majority ensemble of 3 base classifiers in a "one-vs-all" method.

## Files
### FraudedRawData
The directory FraudedRawData contains 40 users’ history bash commands segments. <br>
For every UserX (X=1..40), you have a file that contains 15,000 bash commands such that each 100 commands are defined as a segment. <br> 
The first 5,000 entries (=50 segments) in each file are training entries, i.e., they are guaranteed to be UserX's commands.  <br>

### challengeToFill.csv
The file challengeToFill.csv provides the key for the rest of the task. <br>
It is a 40x150 matrix. Each row represents a user index and each column represents whether the segment of 100 commands has been entered by the user (labeled by 0) or by a masquerader (1). <br>

### detect.ipynb
running notebook of the code

### Method
This problem is treated as a classification problem. <br>
We used an ensemble model with 3 classifiers in it based on majority voting: GradientBoosting, MLPC (ANN) and Random Forest Classifier. <br> 

TF (term frequency) is used to the entire data to the best option among 1 to 10 ngrams, using Z-scale normalization.  <br>
Next, we converted the results with SVD with 100 features, getting the features to use.  <br>
Finally, we enriched the data with some statistical features, namely mean, std and median.  <br>

#### Architecture
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Deep%20Learning/Detect%20Masqueraders%20Commands/%E2%80%8F%E2%80%8Fclassifiers.PNG)


