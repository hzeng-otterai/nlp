
python -m allennlp.service.server_simple \
    --archive-path ./model/model.tar.gz \
    --predictor textual-entailment \
    -o '{"dataset_reader": {"tokenizer": {"word_splitter": {"type": "spacy"}}}}' \
    --title "Demo of the Paraphrase Identification" \
    --field-name premise --field-name hypothesis