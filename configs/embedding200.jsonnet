{
  "dataset_reader": {
    "type": "DT_reader"
  },
  "train_data_path": 'data/train_t.csv',
  "validation_data_path": 'data/train_v.csv',
  "model": {
    "type": "DT_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "/Users/jinlukang/Downloads/glove.6B/glove.6B.200d.txt",
          "embedding_dim": 200,
          "trainable": false
        }
      }
    },
    "text_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 200,
      "hidden_size": 200,
      "num_layers": 1,
      "dropout": 0.2
    },
    "keyword_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 200,
      "hidden_size": 200,
      "num_layers": 1,
      "dropout": 0.2
    },
    "location_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 200,
      "hidden_size": 200,
      "num_layers": 1,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 1200,
      "num_layers": 2,
      "hidden_dims": [300, 2],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["data_id", "num_tokens"], ["text", "num_tokens"], ["keyword", "num_tokens"], ["location", "num_tokens"]],
    "batch_size": 64
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 10,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}