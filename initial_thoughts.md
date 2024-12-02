# Initial thoughts on the task:
- binary classification problem.
- Text input -- NLP.
- Want to try ML and DL. (need to check the suitable LLM also)

# Step 1: Initial thoughts about the data.

- This is not a text data set  => no need for the Vectorization or embedding.
- Data is not sparsh => So this is not from a Bag of words (BOW) or TF-IDF.
- Data is dense numeric data => So must be from a embedding.
- Shape => (3750, 10000)
- Null value => There are no null values present in the data.
- Imbalance target dataset? =>  Highly Imbalanced dataset. (1: 90%  and -1: 10%)
- Target value is looking like 1, -1.

To Note:

- All the rows are fixed length -- **No need for padding/trimming.**
- 10k features -- **Need to select a suitable Dimensionality Reduction.**
- ***No need to handle null values.***
- Imbalanced dataset: **Need to handle it.**
- **Need to change the target value to 0 and 1, because scikit-learn considers 1 as positive and 0 as negative.**

# Step 2: Flow.

Since the data is embeddings, reduces a lot of work here. (data is pretty clean :) )
The flow should be:
- Split the dataset - to avoid data leakage.
- Re-sampling - over-sampling and under-sampling.

    - resampling only on the train data to keep the test data as original one, maintaining real-world distribution of data.
- Standarise your data
- Then do PCA

    - Why PCA after resampling?, PCA is to get the features covering most variance. If you do PCA before re-sampling the princible components  are computes based on the original data. the re-sampling can affect those principle components.

- Store the pre-processed data in the target folder.
- Model training and hyperparameter tuning.
- Testing and deploying the model.

# Step 2: Choose a dimensionality reduction method.

Have 2 options in mind, PCA and SVD.

- SVD - Would have a been a best choice if the data is a sparse data.
    - why because, the SVD(singular value decomposition) is a method reducing a matrix into 3 matrix and keep only the top given number of singular values. It's easier with the sparse kind of data and can be used with larger dimensionalities since most of them are zero.

- PCA - A method of keeping the features which has the most variance.
    - Best option when we have moderate number of rows (maybe less than 1 lacks normally)
    - Need dense data because of finding the egin values and centering the data.

Note: ***PCA seems to be the better option for me here.***

# Step 3: Handle Imbalance dataset.
- Try re-sampling. Hybrid: Over-sampling + under-sampling + weighted models.
- Give more wight to the minority class.
- Evaluation metric: We can't use Accuracy here. Use Precesion and recall and F1-score. (both FP and FN is important here.)


SMOTE - oversampling not good for the embedding data, because embedding comes with a lot of complex sematic interconnect relationships.
- Start with SMOTE and improve.

**Result: Tested with SMOTE giving bad results**.
- So we can use some GAN (generative adversarial network) models to generate synthetic data.
- we can use pre-trained LLMs here. (but we don't have text data for it.)

# Step 4: Choose model.
Have few models in mind.
- ***Naive Bayes*** -- data is not in categorical/binary formate. The data independent assumption is not here. Not so good to go with this.
- ***SVM*** -- Data will have around 500 - 600 dimensions(not decided yet, has to go under hyperparameter tuning). I can use kernels so might expect a better results.
- ***Logistic regression*** - The data won't pass the assumption check, and I don't want to normalise the data. Can be a baseline model.
- ***KNN*** - suffers with high dimensions, and computationally expensive to get the distance, Not using it.
- ***Tree based models*** -- better to try Random forest and XGBoost.
- ***Nueral network*** --
    - LSTM - since we have huge features, we need to have the long term dependencies of the words in the sentences.
    - Bi-LSTM - It's better to try bi-directional LSTM. Helps in understand the meaning.
- ***Transformers*** - It requires text input, we have embeddings only. So, we can't directly use transformers. Still we can try writing transformers from scratch.

choosed models:
- logistic and SVC - base models.
- Random forest and XGBoost
- Deep learning - LSTM and Bi-LSTM.


# Step 5: Target value fixing.

why this is important?
- We need to choose the evaluation metrics based on the application.
- Take Precision, if you wanna focus more on False Positive(FP). (ie. if you want to reduce FP more.)
- Take Recall, if you wanna focus more on False Negative(FN).

\[
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
\]

\[
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
\]

- In scikit-learn, by default one is considered as positive class and 0 is negative class. precision , recall and f1-score is calculated based on this.
- So it is important to decide which is our positive class and which is our negative class. and mark the labels based on this.

According to this consideration, For us saying spam is positive (our goal).
```
1    90.0
-1    10.0
Name: count, dtype: float64
```
this is our target class distribution, It's not given which encoding represents spam and which encoding represents ham. based on assumption of we have more data as not spam, I am considering 1 as not spam and -1 as spam.

- So. we need to convert -1 to 1 (positive class)
- we need to convert 1 to 0 (negative class).
