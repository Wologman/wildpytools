# Calibration of wildlife classifiers for long term ecological monitoring

## Introduction

This document is to detail some of our learnings working at the Department of Conservation, Predator Science team, building and using various deep learning models (including, but not limited to Alita & Kaytoo), for long term monitoring studies, and also seeing these models deployed in a variety of other settings.


## Use & Choice of Binary Thresholds

Both Kaytoo (for audio classification) and Alita (camera trap classification) are multi-label classifiers.  Essentially they are independently predicting presence or absence of each of the classes. 

For a perfect classifier, the scores for true positives would cluster near 1, and true negatives would cluster near 0.  Any binary theshold around 0.5 would give exactly the same predictions.  

For a more realistic scenario, both these score distributions overlap, but not by very much.  In this scenario the total predictions for any class becomes a function of the threshold choice for that class.  

Even if the scores were compared with each other directly (treating the problem as multi-class rather than multi-label, as is common for camera traps), if there is no calibration of scores between classes then we still create systematic bias.

## Effects of Bias

All deep learning models contain at least some systematic bias related to their training methods and datasets.  This can produce a number of undesirable effects.

* Relative abundance between classes may be inaccurate if the same binary threshold is blindly applied to all, or their prediction scores are compared directly.
* If the models are to be used for monitoring relative changes in wildlife, we may falsely predict trends for the species of interest, based on secondary effects for example wind or rain in the context of audio.  Light conditions, snow or vegetation in the case of images
* Cross correlation between species can lead to the apparent change in one species due entirely to the actual change in some similar looking species.
* Score distributions may be effected by unexpected, or unaccounted for environmental change or species variation between locations.
* The training data may not represent some classes very well, leading to prediction scores in deployment that are too high or low for some default threshold.


## Calibration Methods 

The following was written specifically for Kaytoo and/or Alita, which were both treated as multi-label classification problems.  The argument is similar for all wildlife classification problems.  At a bare minimum, some informed choice of threshold must be chosen.  But for a more interpretable result, we could consider scaling the raw predictions.

### Instance-level scaling
By this I mean maximising the chance of a correct prediction for a single image (or chunk of sound).

A good starting point is would be [Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling) (fit a logistic regression model), with model predictions against ground truth for some pre-existing independent data.  

```P(y=1 | f) = 1 / (1 + exp(A·f + B))```
where *f* is the raw score, *y* is the ground truth (0 or 1), A & B are the slope and offset coefficients.   

Now the scaled scores from different classes can be fairly compared, and a single threshold of 0.5 can be reasonably be used over all classes.  This approach also has the nice outcome of making the scaled model predictions interpretable as an estimate of probability.

If such data is unavailable or is too dissimilar from your study images, a *stratified random* sub-sample of your study data could be used, so long as you can label enough samples for each class of interest for a reasonable fit.

Note that if you do have good historical data to work with, this step is very cheap computationaly and requires no new human annotation time.  For Alita the calibration data can **not** come from DOC's publicly available training data from [Lila Science](https://lila.science/), the model was trained on this.  At the time of releasing of version 3.03 we did not have sufficient independent data to provide a useful collection of slope (A) and offset (B) coefficients.  This will hopefully be available with the next version.

### Calibration for estimating absolute counts

Suppose we require an absolute count for some reason, or need to estimate an uncertainty of that count.  For example we are trying to compare the sensitivity of two different camera setups, or even completely unrelated observation methods like cameras vs footprint tunnels, or build a long term record where running new models over the old data is impractical.

Imagine a histogram of the true positive scores and true negatives.  If we have already scaled as above for instance level predictions the theshold should be on or near 0.5.  However the shapes of the distributions have no reason to be symmetrical, meaning the area of false positives and false negatives will not cancel each other.  That difference in area is equal to the error in the absolute count.  

Also it is entirely likely that our localised study images are subtly different in some unexpected way to those of the stored data used for the scaling, potentially adding to any asymmetry.  For example we're trying to count rabits, but in a loction where there are many hares present.  This may affect the shape of the rabit prediction score distribution.

To correct for the above errors, we could take a random sub-sample of the study data, then iteratively apply a temperature scaling step, or adjust the threshold slightly until the error is zero for that sub-sample. The advantage of temperature scaling rather than threshold adjustment is that you can leave the final threshold at 0.5 across all classes, and just sum up the 1s to get an estimate of the total, if that is how you aggregated when performing the scaling step.  So the scores remain interperatable as something like probability.

To go a step further and give an agregate prediction with confidence intervals, you could implement bootstrapping with your random sub-sample.  

Now you have a solid estimate of your total, plus an uncertainty, based on a small sub-sample of your actual study data, with the model applied to the rest.  In doing so you have corrected for both internal model bias, and effects of any unexpected localised behaviour that may be occuring with this particualar season/location/camera setup/species variant etc.

### Correction for dynamic changes in class distribution within the monitoring period

The above two steps are 'static',  in that they considers totals over some fixed period of time with a fixed real-life distributioin between the classes.  The risk here is that some class (for example Tui for the audio context) may be increasing, but the bird you are interested in sounds/looks similar (for example Bellbird). If the confusion between these two birds is not considered then there may be a false 'increase' in Bellbirds detected, whilst their population was stable.

Though this is a second order effect, it is often these changes that are of most interest to the research question. To correct for this a confusion matrix between the relative classes could be applied to any relative changes from the step 2 counts.  Note that if the confusion matrix is available and covers the relevant classes this is a computationally cheap step, that could be automated in any workflow.

## Active learning | Build a one-off correction model for a particular project

Strictly speaking this isn't calibration, though it also achieves the goal of reducing error by adapting to the domain at hand.

The idea is to build a local correction model for the predictions on top of the original model, to improve the scores and adapt the predictions to their current environment, thus reducing the absolute error.   The metric is maximum information improvement for given amount of expert labelling effort.

Future versions of both Kaytoo and Alita will do something like output the backbone embedding vector for each image or audio segment. In the case of audio, we may involve the attention weights in tis as well, and consider the differing results from multiple backbones.  This will let us build a small correction model in-situ on a per-project basis.  It takes us days on a high performance machine to train the full deep learning model from scratch,  but only minutes or seconds to train a small embeddings based model on a few hundred samples.

The embeddings will be used together with some clustering and sub-selection decisions to identify the most informative samples for human-in loop labelling.  Once those samples are labelled the correction model can then be trained retrained.  This process continues iteratively in small batches until there are no more outliers or scores near the decision threshold.  

This approach reduces the problem overlapping prediction scores by learning from and correcting for local features, greatly reducing the size of any error for a modest amount of human-in-loop annotation.  In the case of audio, additional measured features (for example background noise, or detection of rain) could be added at the same stage.

The added bonus of this step is that we can store those very informative samples for future training of the whole model, thus helping future versions of the base model to generalise to new domains.

## References