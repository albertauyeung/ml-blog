# 001 - Understanding Accuracy: The Basic Metric in Machine Learning

When we want to talk about the performance of a [machine learning](https://en.wikipedia.org/wiki/Machine_learning) model, **accuracy** is often the first metric that comes to mind. This is probably the most commonly used metric. In this blog post, we will discuss what accuracy is, when to use it, and when to be cautious.

## What is Accuracy? 🎯

The term "accuracy" tends to be used in a more general sense in daily life. For example, one might say that a clock is accurate if it shows the correct time; or one might say that the weather forecast was accurate if it predicted the temperature correctly (or with little error).

However, in the context of machine learning, accuracy is a specific metric that measures how well a model performs on a given dataset. In addition, it is a metric for **classification problems**, where the goal is to predict **a class label** 🏷️ (e.g., spam or not spam, fraud or not fraud) for a given input.

In mathematical terms 📐: 

$$
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

Or more formally:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Where:
- **TP** = True Positives (positive samples correctly predicted)
- **TN** = True Negatives (negative samples correctly predicted)
- **FP** = False Positives (negative samples incorrectly predicted as positive)
- **FN** = False Negatives (positive sampoles incorrectly predicted as negative)

## Example: Image Classification 🖼️

Imagine you’ve built a model to detect whether a given image is a picture of a cat 🐱. You test it on a dataset of 1,000 images where:
- 600 are not cats (negative)
- 400 are cats (positive)

Let’s say our model classifies 550 of the 600 non-cat images correctly (true negatives) and 360 of the 400 cat images correctly (true positives). The accuracy is:

$$
\text{Accuracy} = \frac{550 + 360}{1000} = \frac{910}{1000} = 0.91 \text{ or } 91\%
$$

## Computing Accuracy in Python

Based on the formula above, we can calculate accuracy by dividing the number of correct predictions by the total number of predictions. For example:

```python
y_true = [0, 1, 1, 0, 1]  # True labels
y_pred = [0, 1, 0, 0, 1]  # Predicted labels

correct_predictions = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp])
total_predictions = len(y_true)

accuracy = correct_predictions / total_predictions
print(f'Accuracy: {accuracy:.2f}')
# Output: Accuracy: 0.80
```

We can also easily calculate accuracy using Python libraries like `scikit-learn`:

```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1]  # True labels
y_pred = [0, 1, 0, 0, 1]  # Predicted labels

accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.2f}')
# Output: Accuracy: 0.80
```

## When (Not) to Use Accuracy ✅

Accuracy is simple to calculate and intuitive to understand, making it a popular metric. However, it’s not always the best choice. It can be misleading in certain situations.

Let's consider the above example again. If the test set had 950 non-cat images and only 50 cat images, a model that always predicts “not cat” would have an accuracy of 95%. Even though the accuracy is high, the model completely fails at detecting cats. This example shows that when one of the classes is dominant, accuracy can be misleading (see [accuracy paradox](https://en.wikipedia.org/wiki/Accuracy_paradox)).

The **key takeaway** here is that accuracy should be used cautiously in cases of **imbalance datasets**.

## Other Metrics to Consider 📊

When accuracy is not a reliable metric, we can use some other metrics such as:
- **Precision**: Measures how many of your positive predictions were actually correct.
- **Recall**: Looks at how many actual positives your model successfully detected.
- **F1-Score**: The harmonic mean of precision and recall, balancing the two.
- **AUC-ROC**: Evaluates the trade-off between true positive rate and false positive rate.
- **Confusion Matrix**: A table that shows the detailed breakdown of your model’s predictions.

We will explore these metrics in future posts!

## Final Thoughts 💭

Accuracy is a great starting point when evaluating a model, but it's not the whole picture. It works well when your data is balanced, but when it's not, other metrics like precision, recall, and the F1-score can give you deeper insights into your model’s performance.
