# python program equivalent to the following command
# allennlp evaluate /path/to/model.tar.gz https://s3-us-west-2.amazonaws.com/allennlp/datasets/quora-question-paraphrase/test.tsv

import sys
import logging
import glob
import os.path as op
import json

from allennlp.commands import main

if __name__ == "__main__":
    # getting the last updated model
    files = list(filter(op.isfile, glob.glob("./output_*/model.tar.gz")))
    files.sort(key=lambda x: op.getmtime(x))
    model_path = files[-1]
    
    # getting the test file from the config file
    config_path = op.join(op.dirname(model_path), "config.json")
    with open(config_path, "r") as config_file:
        config_json = json.load(config_file)
    test_file = config_json["test_data_path"]

    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "evaluate",
        model_path,
        test_file,
        "--include-package", "my_library",
        "--cuda-device", "0"
    ]

    logging.basicConfig(level=logging.INFO)
    
    sys.path.append('.')
    main()
