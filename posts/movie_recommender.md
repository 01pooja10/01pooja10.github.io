With the evolution of machine learning, very little feels impossible. And so when we feel like we've watched every movie we think we'll enjoy, machine learning comes to the rescue with the concept of recommender systems.

Various content-streaming giants like Netflix, YouTube, Spotify, etc. make use of such recommender systems to effectively suggest movies/shows/songs to the users depending on their interests and ratings.

So what does a recommendation engine do? Based on an input (name of a movie) provided by the user, the model will pick out certain other titles depending on its intuition (which is nothing but math) which will be displayed as recommendations for that input. Let's further explore how a model picks out similar movies.

For this we'll use 2 algorithms:

- **Content-based filtering** — Different attributes like plot, genre, cast & crew information, movie description, etc. are used to create something called the similarity matrix consisting of all the movies arranged in an order.
- **Collaborative filtering (user-based)** — The Nearest-Neighbors algorithm is used to pick similar movies (using the cosine similarity metric) from the cluster depending on how similar they are.

We'll be exploring both these algorithms in detail so read on to gain more insight into the working of the 2 algorithms and how they perform in real-time.

---

## Content-based filtering

As the title reveals, this algorithm makes use of the actual features/content of the movie i.e. movie name, cast, genre, etc. to make customized predictions for the user.

The dataset is first cleaned and a separate column consisting of the various details related to the movie is created using different features like genre, cast, title, plot, etc. This column is then passed through the **Tf-Idf vectorizer**. The Term Frequency-Inverse Document Frequency is used to convert English words to vectors. This is done based on:

1. **Frequency of occurrence** of a given word in one document — Term Frequency (Tf).
2. **How frequently the same word** is used in the entire "corpus" i.e. how significant it is, with respect to the entire dataset — Inverse Document Frequency (Idf).

Let's proceed and build a similarity matrix using these Tf-Idf vectors. For building this similarity matrix, we use the **linear kernel separation**. This is especially useful in case the dataset consists of many features as it is faster and most of the text-related features are linearly separable.

This in turn becomes the similarity (reference) matrix for our model to pick out suitable recommendations for the movie input.

We then define a function that will take in a movie's name as the input, find similar titles from the similarity matrix and display the best possible recommendations. This means that our model will pick out 10 elements with similarity scores in the same range as that of the input movie's value in the similarity matrix.

We eventually correlate the names of the movies with the indices of the element picked in the similarity matrix to fetch all the similar movies in the list. This list is sliced to contain the first 10 movies (excluding the input itself) and is displayed to the user.

The recommendations are decent and actually correlate to the movie entered. This shows how valuable the algorithm is, with respect to actually retaining the contextual information of the movie.

### Drawbacks of content-based filtering

Content-based filtering requires users to input only those movies that are already present in the database. It requires the exact title with spaces, punctuations, and proper spelling. If the movie is unavailable, it doesn't display recommendations.

---

## Collaborative filtering (User-based)

User-based collaborative filtering is a popular alternative to content-based filtering since it makes use of various users' ratings that might be similar to our interests instead of using just the plot or the genre of the movie.

Firstly, the movie dataset is read into a variable and processed. Another dataset for user-ID and ratings is read into a different variable. The aim is to combine both these tables into one, based on their respective movie-ID columns. Once this is done, a few duplicate rows and null values can be dropped out of the combined dataset.

Secondly, a **pivot table** is used to depict *users' ratings* in accordance with the *movie titles* arranged with respect to the *user-ID*. This helps us extract important information from similar users (their ratings) instead of using the movie's plot, genre, or such semantic details. In its essence, the pivot table provides a summary of the entire dataset by correlating each of the 3 important aforementioned features i.e. user ratings, movie titles, and user-ID.

Many entries in the pivot table are zeros since such a vast number of movies cannot possibly be rated by every user. Due to the unused state of so many rows and columns i.e. the presence of so many null values, a simpler compressed version of the same matrix can be created to accelerate the entire process. This table, with 9719 rows and 384 columns, can be neatly compressed using a functionality called the CSR matrix.

The pivot table is converted into a **Compressed Sparse Row (CSR) matrix** using Python's scipy library. A [CSR matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html), as the name suggests, is useful in compressing a high-dimensional matrix such as our pivot table into one that is an encoded representation of the same in lower dimensions.

Further, an algorithm called **Nearest-Neighbors** is used to arrange similar movies into groups and then pick out a few titles (ten movies, in our case) from them according to their closeness to the chosen movie. The purpose of using such an algorithm is that, as different users watch and rate various movies, those with interests similar to the user seeking recommendations, will be picked as the nearest neighbors and their preferred movie choices will be recommended with relation to the CSR matrix.

To arrange similar movies in groups, a similarity metric is used by the model. In our case, the metric chosen is **cosine similarity**. Cosine similarity, unlike our other options (Euclidean distance, Manhattan distance, etc.), is preferred because it takes into consideration the angular separation between 2 movie vectors instead of solely calculating the distance between them.

For example, if the angle between 2 movies is 0 degrees, the movies are very similar else if it is 90 degrees, the movies are highly unrelated.

In this way, the top 10 recommendations are chosen and displayed to the user based on their input. But to actually get recommendations for user input i.e. a movie name, we need to compare the latter to pre-existing movies in the database so that the model can process it and produce outputs (10 movies) based on the CSR matrix. For this comparison, we use the [fuzzywuzzy](https://pypi.org/project/fuzzywuzzy/) library in Python for string matching.

This library uses the **Levenshtein distance**, which is the distance between 2 string sequences. This metric depends on how many edits (changes with respect to individual letters) need to be made to each letter in a given string to make it a counterpart of the string it is being compared to. This is how similarity between strings is calculated.

The values extracted from the fuzzywuzzy function — extractBests() are inserted into a list and represent the indices of the CSR matrix. These indices correspond to similar movies stored in the matrix. We then use the k-neighbors function with 10 as the value for the number of recommendations to obtain the same.

### Drawbacks of (user-based) collaborative filtering

Collaborative filtering makes neither the expected nor any personalized recommendations. It doesn't have any clue as to what genre the movie belongs to or why the user likes that particular movie. It merely follows the prescribed mathematical guidelines to establish and print recommendations.

> The code to the working application (via GitHub) can be found [here](https://github.com/srm-mic/Movie-Recommender-System). Do visit the deployed version of the application to get some customized movie recommendations and try it out for yourself.

---

## Conclusion

In this blog, we have learnt about, implemented, and tested 2 (content-based and user-based collaborative filtering) algorithms that can efficiently and accurately recommend movies to users based on their input. We also explored methods to preprocess datasets consisting of information about various movies and the different techniques that can be employed to successfully obtain recommendations.
