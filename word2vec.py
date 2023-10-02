import matplotlib.pyplot as plt
import joblib
import numpy as np
import re
from collections import Counter
import tqdm

def create_dataset():
	"""
	Combines the two txt files inside ds2_data folder. Also, discads all the non utf-8 characters.

	"""
	with open('ds2_data/interview_ds.txt', 'r', encoding='utf-8') as file:
	    content_1 = file.read()

	with open('ds2_data/interview_ds_2.txt', 'r', encoding='utf-8') as file:
	    content_2 = file.read()

	combined_data = content_1 + '\n' + content_2

	with open('data.txt', 'w', encoding='utf-8') as f:
	    f.write(combined_data)


def initialize_vectors(dim, vocab_len):
    """
    Initialize the matrices using random samples from a normal distribution.

    Args:
    dim (int): The length of the dimensions for each word vector.
    vocab_len (int): The number of words in the vocabulary.

    Returns:
    np.ndarray: Initialized word vectors for each word in the vocabulary.
    """

    scale_factor = 100 / np.sqrt(vocab_len * dim)
    C = np.random.randn(vocab_len, dim) * scale_factor

    return np.matrix(C)


def one_hot(word, vocabulary):
    """
    Creates a one-hot encoded vector for a given word based on the provided vocabulary.

    Parameters:
    word (str): The input word to be one-hot encoded.
    vocabulary (List[str]): The list of words that makes the vocabulary.

    Returns:
    tuple: A tuple containing the one-hot encoded numpy array and the index of the word in the vocabulary.
           If the word is not in the vocabulary, returns a zero vector and -1 as the index.
    """

    vocab_len = len(vocabulary)
    ret = np.zeros(vocab_len, dtype=np.uint8)
    try:
        index = vocabulary.index(word)
        ret[index] = 1
        return ret, index
    except ValueError:  # word is not in vocabulary
        return ret, -1

def negative_distribution():
	"""
	This function samples negative words from the vocabulary where the probability
    of each word is determined by the following ratio:

        count(word) / |Vocabulary|^(3/4)

    The table used for sampling is assumed to be a pre-existing global variable,
    populated according to the above-mentioned probability distribution.

    The table_size is also considered as a global variable defining the size of the table.

    Returns:
        int: The index of the sampled word from the table.
	"""

	return table[np.random.randint(low=0, high=table_size, size=1)][0]

def sigmoid(z, func = "logistic"):
	"""
    Computes the value of a sigmoid function, given an input `z` and the type of sigmoid function.

    Parameters:
    z : float or np.array
        The input to the sigmoid function, can be a single value or a numpy array.

    func : str, optional, default="logistic"
        The type of sigmoid function to compute. The supported values are "logistic" and "tanh".
        - If "logistic", the standard logistic sigmoid function is computed.
        - If "tanh", the hyperbolic tangent sigmoid function is computed.

    Returns:
    float or np.array
        The computed value of the sigmoid function. If `z` is a numpy array, the returned value will also be a numpy array, where each element corresponds to the sigmoid of the corresponding element in `z`.

    """

	if func == "logistic":
		return 1/(1 + np.exp(-z))
	elif func == "tanh":
		return (np.exp(-z) - np.exp(-z))/(np.exp(-z) + np.exp(-z))

def train(words, window = 2, hidden_units = 300, epoch = 1):

	"""
    Trains word embeddings using a simple neural network.

    The function goes through the list of words, and for each word (centre word),
    it considers the words within the specified window size around it (outside words).
    It then optimizes the word embeddings to predict the outside words given the centre word.

    :param words: List of strings. The corpus of text represented as a list of words.
    :param window: Integer, optional, default is 2. The number of words to consider to the left and right
                   of the centre word within which predictions are made.
    :param hidden_units: Integer, optional, default is 300. The size of the word embeddings, or the number of
                         units in the hidden layer of the neural network.
    :param epoch: Integer, optional, default is 1. The number of times to iterate over the entire corpus of words.

    :return: U: numpy.ndarray. The word embeddings learned for each word in the vocabulary.
	"""

	U = initialize_vectors(hidden_units, vocab_len)
	V = initialize_vectors(hidden_units, vocab_len)

	iters = 0
	for iters in range(epoch):
		# print(iters)
		for i,word in tqdm.tqdm(enumerate(words), total = len(words)):
			# print word
			#if the centre word is a punctuation go to the next word
			if re.search(regex, word):
				continue

			else:
				for j in range(-window, window+1):
					centre_word = word
					try:
						outside_word = words[i+j]

					except IndexError:
						continue

					#if the outside_word or the centre_word is a punctuation break the loop and go to next outside word
					if re.search(regex, outside_word):
						print(outside_word)
						break
						# continue

					#Maximise the probability for the centre_word and outside_word
					x1, x2 = one_hot(centre_word, vocabulary), one_hot(outside_word, vocabulary)
					k = 20
					learning_rate = 0.1
					vc, vw = x1[0]*U, x2[0]*V

					first_term = sigmoid(np.multiply(vc, vw))
					derivatives = []
					for i in range(k):
						index_of_negative_sample = negative_distribution()
						if re.search(regex, vocabulary[index_of_negative_sample]):
							continue
						else:
							vn = V[index_of_negative_sample]
							value_neg = sigmoid(np.multiply(vc, vn))
							temp = np.multiply(value_neg, vc)
							derivatives.append(temp)
							V[index_of_negative_sample] = V[index_of_negative_sample] - learning_rate * temp

					derivatives.append(np.multiply((first_term - 1), vc))

					#This the objective function that needs to be minimised
					# E = -log(first_term) - log(second_term)

					error_derivatives_positive = np.multiply((first_term - 1), vc)
					error_derivatives_negative = sum(derivatives)

					V[x2[0]] = V[x2[0]] - learning_rate * error_derivatives_positive
					U[x1[0]] = U[x1[0]] - learning_rate * error_derivatives_negative

		iters+=1

	return U

def build_table(distribution):
	"""
    Builds a table based on a given distribution.

    This function creates a NumPy array (`table`) initialized with zeros,
    and fills it with indices based on the given `distribution` array.
    The range that each index of the distribution fills in the table is
    proportional to its value in the `distribution`.

    Parameters:
    - distribution (list or np.array): A list or a NumPy array representing
      the distribution. Each element in the `distribution` represents the
      proportion of the corresponding index in the resulting `table`.
      The sum of all elements in `distribution` should ideally be 1.

    - table_size (int, optional): The size of the resulting table.
      Defaults to int(1e8).

    Returns:
    - np.array: A NumPy array of type `np.int16`, representing the built
      table, with the size of `table_size`.
	  """

	table = np.zeros(table_size, dtype=np.int16)
	previous_j = 0
	for i, value in enumerate(distribution):
	    j = int(table_size * value)
	    table[previous_j:previous_j + j] = i
	    previous_j = previous_j + j
	return table

if __name__ == '__main__':
	regex = "[^a-zA-Z0-9]"

	# Read the data and convert it to a list of words
	with open('data.txt', 'r', encoding='utf-8') as file:
		text = file.read()
	text = text.lower()

	clean_text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

	words = clean_text.split()
	count = Counter(words)

	vocabulary = list(count.keys())  # List of all the words in the vocabulary
	vocab_len = len(vocabulary)

	freq = np.float32(list(count.values()))
	numerator_dist = np.power(freq, 0.75)  # Frequency raised to the power 3/4 which is empirical
	norm_term = sum(numerator_dist)  # Normalising factor
	distribution = numerator_dist / norm_term

	table_size=int(1e8)
	table = build_table(distribution)

	word_representations = train(words)


	word_dict = {}
	for i, word in enumerate(vocabulary):
		word_dict[word] = word_representations[i]

	joblib.dump(word_dict, 'words.pkl', compress = 3)
