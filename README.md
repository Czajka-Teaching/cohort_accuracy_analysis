# Semester project repository

**Notes:**
- Please provide your name (or netID) here so we can identify whose repository it is: ...
- Teams: please indicate your individual contributions in each report

## Deliverables (copied from Canvas)

### Deliverable #3: First solution + validation set accuracy

Push to the project GitHub repo your first solution designed and evaluated with train and validation partitions of your data. This should include:

1. **Source codes** with instructions how to run your trained neural network on a single validation sample (please attach it to your solution). We should be able to run your programs **without any edits** -- please double check before submission that the codes are complete and all the package requirements are listed **(8 points)**.
2. **A report** (as a `readme` in GitHub) with:
- A short justification of your neural network architecture (number and type of layers, loss function, optimization algorithm, etc.) and -- if that applies -- non-NN feature extractor **(3 points)**
- A classification accuracy achieved on the training and validation sets. That is, how many samples were classified correctly and how many of them were classified incorrectly (It is better to provide a percentage instead of numbers). More advanced students (especially those attending the 60868 section of the course) can select the performance metrics that best suit their given problem (for instance, Precision-Recall, f-measure, plot ROCs, etc.) and justify the use of the evaluation method **(3 points)**
- A short commentary related to the observed accuracy and ideas for improvements (to be implemented in the final solution). For instance, if you see almost perfect accuracy on the training set, and way worse on the validation set, what does it mean? Is it good? If not, what do you think you could do to improve the generalization capabilities of your neural network? **(6 points)**


### Deliverable #4: Final solution + test set accuracy

Push to the project GitHub repo your final solution tested on the test partition of your data. This should include:

1. **Source codes** with instructions how to run your final solution on a single test sample (please attach it to your solution). We should be able to run your programs **without any edits** -- please double check before submission that the codes are complete and all the package requirements are listed **(6 points)**.
2. **A report** (as a `readme` in GitHub) with:
- A description of the test database you collected or downloaded: What is the size of the database? What is different when compared to the training and validation subsets? Why you believe these differences are sufficient to test the generalization capabilities of your final programs? **(3 points)**
- A classification accuracy achieved on the test set. Use the same metrics as in deliverable #3 **(1 point)**
- Most of you should see worse results on the test set when compared to the results obtained on train/validation sets. You should not be worried about that, but please provide the reasons why your solution performs worse (with a few illustrations, such as pictures or videos, what went wrong). What improvements would you make to lower the observed error rates? **(5 points)**

