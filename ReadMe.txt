MLmodel2.py can be run through the command line.

For running MLmodel2.py for objective 1 to run two predictive models
(SVC, Logisitic Regression) use the following command line arguments
MLmodel2.py -KFeatures False 

For running varying features 100 through 561 use the command line arguments
-KFeatures True -start 100 -stop 561 -o 'Accuracy_Features.csv' and the results for 
accuracy from each features are recorded in the 'Accuracy_Features.csv'

