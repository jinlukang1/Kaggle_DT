allennlp predict \
  --output-file ./DT_predict.json \
  --include-package tagging \
  --predictor DT_predictor \
  --use-dataset-reader \
  --silent \
  /data1/jinlukang/DT_output/baseline data/test.csv