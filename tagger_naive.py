# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import numpy


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")

    output = open(output_file, "w")
    training_table = {}

    # Training
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
        training.close()

    # Output
    test = open(test_file, "r")
    for line in test:
        word = line.strip()
        tag_tracker = 0
        best_tag = ""
        if word not in training_table:
            continue
        for tag in training_table[word]:
            if tag_tracker < training_table[word][tag]:
                tag_tracker = training_table[word][tag]
                best_tag = tag
        output.write(word + " : " + best_tag + "\n")
    test.close()
    output.close()


def create_initial_prob(training_table, total_pairs):
    prob_table = {}
    for tag in training_table:
        num_words = len(training_table[tag])
        prob_table[tag] = num_words/total_pairs
    return prob_table


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
