
#!/usr/bin/python

import sys
import pickle

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'deferred_income', 'restricted_stock_deferred', 'director_fees'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# Remove outliers based on outlier investigation
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']

# https://stackoverflow.com/questions/5844672/delete-an-item-from-a-dictionary - Used as source to do data cleaning
# See Sources document for further details.

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for a in my_dataset:  # creating new feature
    if type(my_dataset[a]['salary']) != int:
        my_dataset[a]['salary_bonus'] = 0
    elif type(my_dataset[a]['bonus']) != int:
        my_dataset[a]['salary_bonus'] = 0
    else:
        my_dataset[a]['salary_bonus'] = my_dataset[a]['bonus'] + my_dataset[a]['salary']

for a in my_dataset:  # creating new feature
    if type(my_dataset[a]['restricted_stock_deferred']) != int:
        my_dataset[a]['resto_dirfees'] = 0
    elif type(my_dataset[a]['director_fees']) != int:
        my_dataset[a]['resto_dirfees'] = 0
    else:
        my_dataset[a]['resto_dirfees'] = my_dataset[a]['restricted_stock_deferred'] + my_dataset[a]['director_fees']


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
print test_classifier(clf, my_dataset, features_list) # Using Udacity provided validation code using Stratified Shuffle Split



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)