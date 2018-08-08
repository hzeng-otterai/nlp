# allennlp evaluate /path/to/model.tar.gz --evaluation-data-file (https://s3-us-west-1.amazonaws.com/handsomezebra/public/Quora_question_pair_partition.zip)#Quora_question_pair_partition/test.tsv

import sys
import logging
import glob
import os

from allennlp.commands import main

# getting the last updated model
files = list(filter(os.path.isfile, glob.glob("./output_*/model.tar.gz")))
files.sort(key=lambda x: os.path.getmtime(x))
model_path = files[-1]

if "snli" in model_path:
    test_file = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_test.jsonl"
elif "quora" in model_path:
    test_file = "(https://s3-us-west-1.amazonaws.com/handsomezebra/public/Quora_question_pair_partition.zip)#Quora_question_pair_partition/test.tsv"


# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "evaluate",
    model_path,
    test_file,
    "--include-package", "hznlp",
    "--cuda-device", "0"
]

logging.basicConfig(level=logging.INFO)

main()
