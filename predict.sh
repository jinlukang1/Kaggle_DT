allennlp predict \
  --output-file ./DT_predict.json \
  --include-package tagging \
  --predictor DT_predictor \
  --use-dataset-reader \
  --silent \
  /data1/tagging/DT_output_dirs/submit data/test.csv