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

To Note:

- 1 | all the rows are fixed length -- **No need for padding/trimming**
- 2 | 10k features -- **Need to select a suitable Dimensionality Reduction**
- 3 | ***No need to handle null values***.
- 4 | Imbalanced dataset: **Need to handle it.**

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
- 3 | Evaluation metric: We can't use Accuracy here. Use Precesion or recall and F1-score.

# Step 4: Choose model.
Have few models in mind.
- 1 | Naive Bayes -- data is not in categorical/binary formate. The data independent assumption is not here. Not so good to go with this.
- 2 | SVM -- Data will have around 500 - 600 dimensions(not decided yet, has to go under hyperparameter tuning). I can use kernels so might expect a better results.
- 3 | logistic regression - The data won't pass the assumption check, and I don't want to normalise the data. But I can try normalise it and check the results. Can be a baseline model.
- 4 | KNN - suffers with high dimensions, and computationally expensive to get the distance, Not using it.
- 5 | Tree based models -- better to try Random forest and XGBoost.
- 6 | Nueral network. - of course.
