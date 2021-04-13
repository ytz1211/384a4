# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import time
import numpy
import pandas


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")

    output = open(output_file, "w")
    training_table = {}
    training_pairs = []

    # Training files being read
    for training_file in training_list:
        training = open(training_file, "r")
        for line in training:
            word_tag = line.split(" : ")
            word = word_tag[0].strip()
            tag = word_tag[1].strip()

            training_pairs.append((word, tag))
        training.close()

    # Create list of all the unique tags seen
    tag_set = []
    for pair in training_pairs:
        if pair[1] not in tag_set:
            tag_set.append(pair[1])

    # Create transition matrix
    start = time.time()
    transition_matrix = create_transition_probs(training_pairs, tag_set)
    print("done transition matrix")

    # Create emission matrix from test words
    test_words = []
    test = open(test_file, "r")
    for line in test:
        word = line.strip()
        test_words.append(word)
    test.close()

    word_tag_sequence = viterbi(
        test_words, training_pairs, tag_set, transition_matrix)

    # Output
    for pair in word_tag_sequence:
        output.write(pair[0] + " : " + pair[1] + "\n")
    output.close()
    end = time.time()
    print("transition matrix and viterbi time taken in seconds is " + str(end-start))


def create_transition_probs(training_pairs, tag_set):

    # Initialize tag transition matrix
    tag_matrix = numpy.zeros((len(tag_set), len(tag_set)), dtype='float32')

    # Load values into tag transition matrix
    for i, t1 in enumerate(tag_set):
        for j, t2 in enumerate(tag_set):
            t1_count = 0
            for pair in training_pairs:
                if pair[1] == t1:
                    t1_count += 1
            t2t1_count = 0
            for k in range(len(training_pairs)-1):
                if training_pairs[k][1] == t1 and training_pairs[k+1][1] == t2:
                    t2t1_count += 1
            tag_matrix[i][j] = t2t1_count/t1_count

    # Return transition matrix
    return tag_matrix


def create_emission_prob(words, tag_set, training_pairs):

    # Initialize emission matrix
    emission_matrix = numpy.zeros((len(tag_set), len(words)))

    # Load values into emission matrix
    for i, tag in enumerate(tag_set):

        matching_tag_pairs = []
        for pair in training_pairs:
            if pair[1] == tag:
                matching_tag_pairs.append(pair)
        tag_count = len(matching_tag_pairs)

        for j, word in enumerate(words):

            matching_words = 0
            for pair in matching_tag_pairs:
                if pair[0] == word:
                    matching_words += 1

            emission_matrix[i][j] = matching_words/tag_count

    return emission_matrix


def create_initial_prob(tag_set, transition_matrix):
    # Initial prob is thought to be from a period (punctuation)
    initial_probs = {}
    tags_df = pandas.DataFrame(
        transition_matrix, columns=list(tag_set), index=list(tag_set))
    for tag in tag_set:
        initial_probs[tag] = tags_df.loc['PUN', tag]
    return initial_probs


def viterbi(words, training_pairs, tag_set, transition_matrix):
    state = []
    state_probabilities = []
    emission_matrix = create_emission_prob(words, tag_set, training_pairs)
    initial_probs = create_initial_prob(tag_set, transition_matrix)

    for key, word in enumerate(words):
        # Produce list of probabilities for observations
        state_probabilities.clear()
        for t, tag in enumerate(tag_set):
            # Initial prob is thought to be from a period (punctuation)
            if key == 0:
                transition_probability = initial_probs[tag]
            else:
                transition_probability = transition_matrix[t-1, t]

            # Emission and state probabilities
            emission_probability = emission_matrix[t][key]
            state_probability = emission_probability * transition_probability
            state_probabilities.append(state_probability)

        pmax = max(state_probabilities)
        # Get most likely state for this word
        most_likely_state = tag_set[state_probabilities.index(pmax)]
        state.append(most_likely_state)

    return list(zip(words, state))


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    print("Training files: " + str(training_list))
    print("Test file: " + test_file)
    print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
