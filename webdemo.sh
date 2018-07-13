python -m allennlp.service.server_simple \
    --archive-path ~/model/bimpm_cosine_word.tar.gz \
    --predictor paraphrasing \
    --include-package hznlp \
    --title "Paraphrasing" \
    --field-name premise \
    --field-name hypothesis