# allennlp evaluate /path/to/model.tar.gz --evaluation-data-file (https://s3-us-west-1.amazonaws.com/handsomezebra/public/Quora_question_pair_partition.zip)#Quora_question_pair_partition/test.tsv

import sys
import logging

from allennlp.commands import main

model_path = "./output_20180706T120500/model.tar.gz"
test_path = "./tests/bimpm_test.txt"

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "predict",
    model_path,
    test_path,
    "--include-package", "my_library",
    "--predictor", "bimpm",
]

logging.basicConfig(level=logging.INFO)

main()