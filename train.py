import json
import sys
import logging
import datetime

from allennlp.commands import main

config_file = "experiments/quora_bimpm_cosine_word_char.json"

# Specify overrides
overrides = json.dumps({
	"iterator": {"batch_size": 64},
    "trainer": {"cuda_device": 0}
})

# Specify output dir according to current time
serialization_dir = "./output_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "hznlp",
    "-o", overrides,
]

logging.basicConfig(level=logging.INFO)

main()
