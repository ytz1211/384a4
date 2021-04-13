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
            if word not in training_table:
                training_table[word] = {}
            if tag not in training_table[word]:
                training_table[word][tag] = 0
            training_table[word][tag] += 1
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

    # Viterbi on test words
    test_words = []
    test = open(test_file, "r")
    for line in test:
        word = line.strip()
        test_words.append(word)
    test.close()
    word_tag_sequence = viterbi(
        test_words, training_pairs, tag_set, transition_matrix)
    print("done viterbi")

    # Output
    # test = open(test_file, "r")
    # for line in test:
    #     word = line.strip()
    #     tag_tracker = 0
    #     best_tag = ""
    #     if word not in training_table:
    #         continue
    #     for tag in training_table[word]:
    #         if tag_tracker < training_table[word][tag]:
    #             tag_tracker = training_table[word][tag]
    #             best_tag = tag
    #     output.write(word + " : " + best_tag + "\n")
    # test.close()
    # output.close()

    for pair in word_tag_sequence:
        output.write(pair[0] + " : " + pair[1] + "\n")
    output.close()
    end = time.time()
    print("viterbi time taken in seconds is " + str(end-start))


def create_transition_probs(training_pairs, tag_set):

    # Initialize up tag transition matrix
    tag_matrix = numpy.zeros((len(tag_set), len(tag_set)), dtype='float32')
    # tag_matrix = []
    # for i in range(len(tag_set)):
    #     row = []
    #     for j in range(len(tag_set)):
    #         row.append(0)
    #     tag_matrix.append(row)

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

    # return tag_matrix
    return pandas.DataFrame(tag_matrix, columns=list(tag_set), index=list(tag_set))


def create_emission_prob(word, tag, training_pairs):
    matching_tag_pairs = []
    for pair in training_pairs:
        if pair[1] == tag:
            matching_tag_pairs.append(pair)

    tag_count = len(matching_tag_pairs)

    matching_words = 0
    for pair in matching_tag_pairs:
        if pair[0] == word:
            matching_words += 1

    return matching_words/tag_count


def viterbi(words, training_pairs, tag_set, tags_df):
    state = []

    for key, word in enumerate(words):
        # initialise list of probability column for a given observation
        p = []
        for tag in tag_set:
            if key == 0:
                transition_p = tags_df.loc['PUN', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_p = create_emission_prob(words[key], tag, training_pairs)
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        # getting state for which probability is maximum
        most_likely_state = tag_set[p.index(pmax)]
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
