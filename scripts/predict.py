# python program equivalent to the following command
# allennlp predict /path/to/model.tar.gz /path/to/test_file.txt --predictor textual-entailment


import sys
import logging
import glob
import os.path as op

from allennlp.commands import main

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <test_file> <output_file> <predictor>")
        sys.exit(0)
        
    test_path = sys.argv[1]
    if not op.isfile(test_path):
        print("Test file %s does not exist." % test_path)
        sys.exit(1)
        
    output_path = sys.argv[2]
    
    if len(sys.argv) > 3:
        predictor = sys.argv[3]
    else:
        predictor = None

    # getting the last updated model
    files = list(filter(op.isfile, glob.glob("./output_*/model.tar.gz")))
    files.sort(key=lambda x: op.getmtime(x))
    model_path = files[-1]

    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "predict",
        model_path,
        test_path,
        "--output-file", output_path,
        "--include-package", "my_library",
        "--use-dataset-reader",
        "--silent",
        "--batch-size", "64",
        "--cuda-device", "0"
    ]
    
    if predictor is not None:
        sys.argv += ["--predictor", predictor]

    logging.basicConfig(level=logging.INFO)

    sys.path.append('.')
    main()