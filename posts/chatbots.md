Chatbot is the buzzword when it comes to the field of Natural Language Processing. Many platforms like Amazon, Zomato and our very own [Data Science Community](https://www.dscommunity.in/), make use of a bot for interacting with clients. The medical field has also seen a surge in the usage of chatbots for various emergency reasons. Bots employed for assistance with mental health, first aid procedures, keeping track of appointments, etc. will, at the very least, end up making our lives easier.

In another scenario that isn't entirely impossible, imagine needing quick help with navigating a certain website at 11pm in the night. It isn't exactly conventional for a human to tend to each and every small query you might have that too at such untimely hours. So how convenient will it be if a bot can take over and provide unconditional support?

Building such a bot will be our goal for this blog. Though we may not get close to an exact replica of the IBM Watson Assistant, we'll still end up with something decent that we can, as debutants to the world of chatbots, be proud of. But how does a bot actually process and respond to user inputs? Read ahead to understand its working.

For this blog, we'll be building an Autism awareness bot called **AUSM**. Autism (Autism Spectrum Disorder) is a neuro-developmental disorder with a broad range of conditions and affects various behavioral or social aspects by limiting communicational abilities of those affected by it. The chatbot we build will be able to provide facts, statistics, possible symptoms and related information in an understandable format.

The user will be prompted to ask different questions related to autism and the bot will try to decode them using different functions in order to pick and display the most appropriate response. We will be mainly making use of the **TensorFlow, Keras, NumPy and NLTK** libraries in Python to comprehensively build a fully functioning chatbot.

---

## Loading and understanding the data

Our dataset is going to be a JSON file and is also referred to as the intents file. This will contain all information pertaining to Autism and basic conversational vocabulary. Our job is to parse through the entire file, convert it into a machine-friendly format, train a model on it and finally get some predictions with respect to user input. Seems pretty far-fetched but fear not, you will be able to chat with a bot you built, before you know it.

The intents file will consist of 3 components namely: **tags, patterns and responses**. For example, let's consider the 'greetings' tag; it will contain different possible inputs the user may provide i.e. the patterns and a set of suitable replies that are stored as responses.

- **Tags** correspond to greetings, descriptions, thanks, symptoms, autism and treatment — what the user needs to access in order to pick responses.
- **Patterns** are the possible questions or statements that the user can present to the bot while texting.
- **Responses** are options for possible replies that the bot can pick after decoding the meaning behind user input.

---

## Text preprocessing

Let's begin by tokenizing all the words in the dataset. **Word tokenization** is a functionality provided by the NLTK library and is used to split sentences in our intents file into words for further processing. We then insert the words into lists with respect to their tags.

We iterate through all the contents of our file to extract words and their corresponding tags. They are jointly inserted into a Python list called *x* whereas labels are inserted into a separate list.

These words are then **lemmatized** and converted to lower case for ease of use. Lemmatization is the process of converting words with suffixes to their respective root words — for example, the words "writing", "writes" and "wrote" will be converted to "write" i.e. their base form. This is better than stemming which simply gets rid of the suffix and sometimes leaves us hanging with an incomplete word like "writ" in case of "writing" where the suffix is "-ing".

### Bag of words

We can now implement the popular bag of words method of processing text which is, in its essence, encoding the words into 1s and 0s based on their presence or absence in a sentence to further form sentence vectors.

Let's assume that we've got 2 sentences:

- "This is a pizza."
- "This is not a waffle."

Here the bag of words will take all the words into consideration only once. So these 2 sentences are traversed to get a list of unique words: `["This", "is", "a", "pizza", "not", "waffle"]`

These words are then encoded with 1s for their presence and 0s for their absence in a sentence. So the bag of words representations for both sentences become:

- "This is a pizza": `[1, 1, 1, 1, 0, 0]`
- "This is not a waffle": `[1, 1, 1, 0, 1, 1]`

As you can see, the second sentence "This is not a waffle" isn't in the same order when represented using bag of words. It is instead stored as "This is a not waffle" because the word "pizza" comes in between the words "a" and "not" in the original list of unique words and so, it is assigned 0 due to its absence which in turn changes the order of words in the sentence. This is a drawback when it comes to bag of words.

A for loop is used to iterate over all the lemmatized words in the "patterns" column to check the presence of all possible words in the user inputs. Accordingly, 1s or 0s are appended to a Python list structure called `bag`.

Finally the results of each iteration i.e. the bags for all possible user inputs, combined with their respective labels (appended to the `output_row` list variable) are further appended to a `train` list which is shuffled and split into `train_x` and `train_y` for ease of use while training a neural network.

---

## Model creation and training

For the sake of this blog, we will use artificial neural networks to train our model and the Keras deep learning API to implement the same in code. Inputs will be `train_x` (the encoded pattern vectors) and `train_y` (the encoded output/label vectors), with a batch size of 5.

We'll make use of a vanilla **3 layered neural network with dropout** for training. This in itself should produce good responses after training for 100 epochs. The weights of the model can be saved as a `.h5` file for future use.

Once the training process ends, our model is deemed fit to make predictions for replies to real time user inputs. Let's see how this can be materialized using some functions in Python.

---

## Chatbot's response mechanism

In order to enable our chatbot to respond to real time user input i.e. to see it work in all its glory, we need to include a function that can break the input statement/question down into (bag of words representation) encoded vectors. The sentence also needs to be tokenized and lemmatized.

The user input is firstly sent to the `input_bag()` function for preprocessing. It is then sent to the model, for prediction, in its bag of words form.

The result of the prediction/the first response that the model picks, is stored in a variable as an index number. This then triggers a for loop to search for the particular response in the `intents.json` file. The most appropriate responses are stored in the *resp* list. One random option is picked from the list and is displayed to the user as the result.

Typing "exit" will allow the bot to end the conversation which will otherwise run in an infinite loop.

The chatbot is pretty intuitive and can read our questions/queries to some extent. It can provide considerably relevant answers and can provide quick facts about the topic at hand. It may not have the perfect responses to questions that are very detailed or out of the scope of the data that the bot has been trained on but, for what it's worth, it does provide satisfactory answers when asked questions it can comprehend.

---

## Conclusion

We have successfully built a fully-functioning, artificial neural networks based chatbot with an end-to-end response mechanism. This bot can help clients by answering various questions they might have regarding Autism and its symptoms, signs, statistics, treatment/therapy, etc. The entire process, in its essence, is based on 4 important steps:

1. Loading the data
2. Preprocessing text information
3. Training a neural network on our data
4. Processing real time user inputs to obtain outputs from the model

The same process can be applied to any intents file of your choice and such bots can be built for various purposes related to medicine, movies, sports, etc. The same bot can also be made to generate entirely new responses using something called the **attention mechanism** which learns only what is necessary, from the dataset. And with this, we come to the end of the bot building venture.
