# Normalized Discounted Cumulative Gain (NDCG): A Key Metric for Ranking Systems

When we evaluate the performance of a ranking system, we often use the **Normalized Discounted Cumulative Gain (NDCG)** metric. This metric is widely used in information retrieval and recommendation systems to measure the quality of ranked results.

In this blog post, we will discuss what NDCG is, how it works, and when to use it in practice.

## What is NDCG? 📊

The **Normalized Discounted Cumulative Gain (NDCG)** is a metric used to evaluate the quality of ranked search results. It is particularly useful in scenarios where we need to rank items based on their relevance to a query.

To understand NDCG, let's break down the term:
- **Gain**: The relevance score of an item. It can be binary (relevant or not) or graded (e.g., 1 to 5).
- **Cumulative Gain**: The sum of gains for the top *k* items in the ranked list.
- **Discounted**: The gain is discounted down the list, giving higher importance to items at the top.
- **Normalized**: The cumulative gain is normalized by the ideal ranking to get a score between 0 and 1.

We can try to understand NDCG step by step as follows. We can refer to the following example as we go through the calculation.

### Example: Movie Recommendations 🎥

Imagine you have a movie recommendation system that ranks movies based on user preferences. You have a list of movies and their relevance scores (gains) for a user query. The table below shows the relevance scores for the top 5 movies in a particular ranked list:

| Rank | Movie | Relevance |
|------|-------|-----------|
| 1    | A     | 4         |
| 2    | B     | 2         |
| 3    | C     | 5         |
| 4    | D     | 3         |
| 5    | E     | 5         |

In this example, the relevance scores are graded from 1 to 5, with 5 being the most relevant. The objective of a good recommendation system is to rank the most relevant items at the top. For example, if the user prefers movie E, the system should rank it higher than less relevant movies like D or B.

### Cumulative Gain (CG) 📈

The Cumulative Gain (CG) at position *k* is the sum of the gains of the top *k* items in the ranked list. The formula for CG is given by:

$$
CG@k = \sum_{i=1}^{k} \text{Relevance}(i)
$$

For the example above, the Cumulative Gain at position 3 (CG@3) would be:

$$
CG@3 = 4 + 2 + 5 = 11
$$

### Discounted Cumulative Gain (DCG) 📉

The Discounted Cumulative Gain (DCG) is similar to CG, but it discounts the gain based on the position in the list. The gain of items at the top is given more importance than items further down the list. The discount is determined by the logarithm of the position. The formula for DCG is:

$$
DCG@k = \sum_{i=1}^{k} \frac{\text{Relevance}(i)}{\log_2(i+1)}
$$

Note that because $i$ starts at 1, and $log_2(1) = 0$, therefore we have to add 1 to the position in the denominator to avoid division by zero.

For the example above, the discounted cumulative gain at position 3 (DCG@3) would be:

$$
DCG@3 = \frac{4}{\log_2(2)} + \frac{2}{\log_2(3)} + \frac{5}{\log_2(4)} \approx 7.76
$$

### Ideal Cumulative Gain (IDCG) 🌟

The Ideal Cumulative Gain (IDCG) is the best possible DCG score for a given list. It is calculated by sorting the items by relevance in descending order. The formula for IDCG is:

$$
IDCG@k = \sum_{i=1}^{k} \frac{\text{Relevance}(i)}{\log_2(i+1)}
$$

For the example above, the ideal cumulative gain at position 3 (IDCG@3) would be:

$$
IDCG@3 = \frac{5}{\log_2(2)} + \frac{5}{\log_2(3)} + \frac{4}{\log_2(4)} \approx \approx 10.15
$$

### Normalized Discounted Cumulative Gain (NDCG)

The Normalized Discounted Cumulative Gain (NDCG) is the ratio of the DCG to the IDCG. It normalizes the DCG score by the ideal ranking to get a value between 0 and 1. The formula for NDCG is:

$$
NDCG@k = \frac{DCG@k}{IDCG@k}
$$

For the example above, the normalized discounted cumulative gain at position 3 (NDCG@3) would be:

$$
NDCG@3 = \frac{7.76}{10.15} \approx 0.76
$$

To summarize, we can refer to the following table with all the values calculated at each of the ranks:

| Rank | Movie | Relevance | CG@k | DCG@k | IDCG@k | NDCG@k |
|------|-------|-----------|------|-------|--------|--------|
| 1    | A     | 4         | 4    | 4.00  | 5.00   | 0.80   |
| 2    | B     | 2         | 6    | 5.26  | 8.15   | 0.65   |
| 3    | C     | 5         | 11   | 7.76  | 10.15  | 0.76   |
| 4    | D     | 3         | 14   | 9.05  | 11.45  | 0.79   |
| 5    | E     | 5         | 19   | 10.99 | 12.22  | 0.90   |

## Computing NDCG in Python 🐍

We can calculate NDCG in Python using the following code snippet:

```python
import numpy as np

# Relevance scores for the top 5 movies
relevance = [4, 2, 5, 3, 5]

# Calculate DCG@k
def dcg_at_k(relevance, k):
    return np.sum([
        (relevance[i] / np.log2(i + 2))
        for i in range(0, k)
    ])

# Calculate IDCG@k
def idcg_at_k(relevance, k):
    # Sort the relevance scores in descending order
    ideal_relevance = np.sort(relevance)[::-1]
    return dcg_at_k(ideal_relevance, k)

# Calculate NDCG@k
def ndcg_at_k(relevance, k):
    return dcg_at_k(relevance, k) / idcg_at_k(relevance, k)

# Calculate NDCG@3 for the example
ndcg_3 = ndcg_at_k(relevance, 3)
print(f'NDCG@3: {ndcg_3:.2f}')
# Output: NDCG@3: 0.76
```

We can also use libraries like `scikit-learn` to calculate NDCG ([documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.ndcg_score.html)):

```python
import numpy as np
from sklearn.metrics import ndcg_score

# Relevance scores for the top 5 movies
# The relevance scores should be a 2D array with shape (n_samples, n_ranked_items)
true_relevance = np.asarray([[5, 5, 4, 3, 2]])
scores = np.asarray([[3, 1, 5, 2, 4]])

# Calculate NDCG@3 using scikit-learn
ndcg_3 = ndcg_score(true_relevance, scores, k=3)
print(f'NDCG@3: {ndcg_3:.2f}')
# Output: NDCG@3: 0.76
```

## When to Use NDCG? 🤔

The Normalized Discounted Cumulative Gain (NDCG) is a valuable metric in scenarios where we need to rank items based on their relevance to a query. Here are some common use cases for NDCG:

- **Information Retrieval**: Evaluating search engine results based on user queries.
- **Recommendation Systems**: Ranking items based on user preferences.
- **Document Summarization**: Prioritizing important documents based on relevance.

NDCG is particularly useful when the **relevance** of items is graded or when the ranking of items is crucial for user experience. It provides a comprehensive evaluation of the quality of ranked results, taking into account both relevance and position in the list.

However, when we don't have graded relevance scores or when the ranking of items is less important, other metrics like Precision@K or Mean Average Precision (MAP) may be more suitable.

### Limitations

While NDCG is a powerful metric for ranking systems, it has some limitations:

- **Sensitivity to Position**: NDCG gives more weight to items at the top of the list. This can be a limitation if items further down the ranked list are also important to the user experience.
- **Insensitivity to Missing Items**: NDCG does not account for missing items in the ranked list. If an important item is missing, the NDCG score may not reflect this. To understand the impact of missing items, other metrics such as recall may be more appropriate.
- **Irrelevant Items**: NDCG assumes that all items in the list are relevant (with relevance scores greater than zero). It does not penalize irrelevant items in the list.

## Summary 💭

In this blog post, we discussed the **Normalized Discounted Cumulative Gain (NDCG)** metric, which is used to evaluate the quality of ranked results. While it is a useful metric for ranking systems, it is important to understand its limitations and use it in the right context.
