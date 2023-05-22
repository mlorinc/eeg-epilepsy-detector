# EEG feature extraction optimization using NSGA II

The most usual way how to work with the following program is:

1. Extract features from EEG records using `python src/main.py features:extract 1 12 n`
2. Try to find the most optimal classifier configuration `python src/main.py features:optimize experiments 12`
3. Analyze result from the previous step `python src/main.py features:analyze experiments`

## Brief introduction to optimization procedure

In process of optimization the NSGA II algorithm is utilized to find chromosomes that select which features will be
used in training process of KNN classifier. Every chromosome has 21 genes which each gene is bound to one feature.
To find out which gene represent which feature, please see `evolution.py`.

Every feature is extracted either from PSD or from every epoch which belongs to the sliding window. Therefore, the
largest possible training vector size is `9*window_size*ch_count + 12*window_size*ch_count`. Moreover,
as chromosome contains 21 genes, the search space is 2^21 and that is the reason why NSGA II was chosen.
The EA will not find the most optimal configuration, although it is possible it will find decent
configurations close to optimum.

### Evaluation

Every NSGA II generation chromosomes must be evaluated. To evaluate chromosome, the KNN classifier must be
trained on training data. As the training data contain all features, the unwanted features have to be
filtered out of the training data. Hence, every chromosome contains array of 0 and 1, where 1 stands for _enabled_ and
0 for _disabled_. Based on that mapping, the wanted features are extracted from the training data and are passed to the classifier.

After training phase, test data are used for prediction and for fitness attribute inference. Fitness attributes
consist from:

1. Sensitivity -- how many seizures were detected represented as ratio
2. Latency -- mean duration until the seizure was detected (_real\_start\_time_ - _detection\_time_)
3. Specificity -- false seizure detection count in span of 24 hours

They are compared based on priority, where sensitivity is maximized, latency minimized and specificity minimized.

### Test data

Seizure detection was tested only on patient 12 as this patient has the most seizure events and the classifier
was built on [Shoeb's classifier](https://physionet.org/content/chbmit/1.0.0/shoeb-icml-2010.pdf) which works on single patient only.
The biggest priority was to ensure train data and test data have the same ratio of seizure events, which
is close to 20:20 in that case. Moreover, to ensure experimental data can be used over and over again,
only 1 train/test dataset was created to allow experiment reproduction in the future, despite the
optimization process is not deterministic by its nature.