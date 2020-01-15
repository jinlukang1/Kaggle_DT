allennlp train -f \
    --include-package tagging \
    -s /tmp/tagging/DT_output_dirs/test \
    configs/bert_for_DT_classification.jsonnet