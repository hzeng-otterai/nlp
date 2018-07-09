import json
import sys
import logging
import datetime

from allennlp.commands import main

config_file = "experiments/quora.json"

# Use overrides to train on CPU.
overrides = json.dumps({
    "trainer": {"cuda_device": -1},
    #"vocabulary": {"directory_path": "./temp/vocabulary"}
})

serialization_dir = "./output_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "my_library",
    "-o", overrides,
]

logging.basicConfig(level=logging.INFO)

main()