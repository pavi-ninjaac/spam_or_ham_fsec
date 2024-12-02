# Initial thoughts on the task:
- 1 | binary classification problem.
- 2 | Text input -- NLP.
- 3 | Want to try ML and DL. (need to check the suitable LLM also)

# Step 1: Initial thoughts about the data.

- 1 | Not text data - not need for the Vectorization or embedding.
- 2 | Data is not sparsh - So this is not from a Bag of words (BOW) or TF-IDF.
- 3 | Data is dense numeric data - So must be from a embedding.
- 4 | Shape: (3750, 10000)
- 5 | Null value: There are no null values present in the data.
- 6 | Imbalance target dataset?: Highly Imbalanced dataset. (1: 90%  and -1: 10%)
- 7 | Target value is looking like 1, -1.

To Note:

- 1 | all the rows are fixed length -- **No need for padding/trimming**
- 2 | 10k features -- **Need to select a suitable Dimensionality Reduction**
- 3 | ***No need to handle null values***.
- 4 | Imbalanced dataset: **Need to handle it.**
- 5 | **Need to change the target value to 0 and 1, because scikit-learn considers 1 as positive and 0 as negative**

# Step 2: Flow.

Since the data is embeddings, reduces a lot of work here. (data is pretty clean :) )
- 1 | Dimensionality reduction.
- 2 | Handle imbalance dataset.
- 2 | training and validation.
- 3 | testing.

# Step 2: Choose a dimensionality reduction method.

Have 2 options in mind, PCA and SVD.

- SVD - Would have a been a best choice if the data is a sparse data.
    - why because, the SVD(singular value decomposition) is a method reducing a matrix into 3 matrix and keep only the top given number of singular values. It's easier with the sparse kind of data and can be used with larger dimensionalities since most of them are zero.

- PCA - A method of keeping the features which has the most variance.
    - Best option when we have moderate number of rows (maybe less than 1 lacks normally)
    - Need dense data because of finding the egin values and centering the data.

Note: ***PCA seems the better option for me here.***

# Step 3: Handle Imbalance dataset.
- 1 | try re-sampling
- 2 | Give more wight to the minority class.
- 3 | Evaluation metric: We can't use Accuracy here. Use Precesion and recall and F1-score. (both FP and FN is important here.)


SMOTE - oversampling not good for the embedding data, because embedding comes with a lot of complex sematic interconnect relationships.
- (tested with SMOTE giving bad results).
- So we can use some Deep learning models to generate sinthetic data.
- we can use pre-trained LLMs here. (but we don't have text data for it)

# Step 4: Choose model.
Have few models in mind.
- 1 | Naive Bayes -- data is not in categorical/binary formate. The data independent assumption is not here. Not so good to go with this.
- 2 | SVM -- Data will have around 500 - 600 dimensions(not decided yet, has to go under hyperparameter tuning). I can use kernels so might expect a better results.
- 3 | logistic regression - The data won't pass the assumption check, and I don't want to normalise the data. But I can try normalise it and check the results. Can be a baseline model.
- 4 | KNN - suffers with high dimensions, and computationally expensive to get the distance, Not using it.
- 5 | Tree based models -- better to try Random forest and XGBoost.
- 6 | Nueral network. - of course.

choosed models:
- logistic and SVC - base models.
- Random forest and XGBoost
- Deep learning
- Transformers - It requires text input, we have embeddings only. Still we can try writing transformers from scratch.

# Step 5: Target value fixing.

why this is important?
- We need to choose the evaluation metrics based on the application we need.
- Take Precision, if you wanna focus more on False Positive(FP). (ie. if you want to reduce FP more.)
- Take Recall, if you wanna focus more on False Negative(FN).

\[
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
\]

\[
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
\]

- In scikit-learn, by default one is considered as positive class and 0 is negative class. precision , recall and f1-score is calculated based on this.
- So it is important to decide which is our positive class and which is our negative class. and mark the tables based on this.

According to this consideration, For as saying spam is positive (our goal).
```
1    90.0
-1    10.0
Name: count, dtype: float64
```
this is our target class distribution, It's not given which represents spam and which represents ham. based on assumption of we have more data as not spam, I am considering 1 as not spam and -1 as spam.

- So. we need to convert -1 to 1 (positive class)
- we need to convert 1 to 0 (negative class).


# Improvements:
- Get more data to train the models.
- Try better Data augmentation techniques, like GAN for generate more data.
- Can try fine-tuning LLMs if we have text as input.
- Since we have very less amount of data, even very complex DL models are just overfitting the data. ***Generate or collect more data.***
