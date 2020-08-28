local bert_model = "../models/bert/";

{
    "dataset_reader" : {
        "type": "sentiment-df",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
            }
        },
        "max_tokens": 150
    },
    "train_data_path": "../data/clean_train.csv",
    "validation_data_path": "../data/clean_valid.csv",
    "model": {
        "type": "sentiment_extractor",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        },
        "encoder": {
            "type": "pass_through",
            "input_dim": 768
        }
    },
    "data_loader": {
        "batch_size": 48,
        "shuffle": true
    },
    "validation_data_loader": {
        "batch_size": 48,
        "shuffle": false
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 20,
        "patience": 5,
        "cuda_device": 3,
        "serialization_dir": './run/'
    },
//    "distributed": {
//        "cuda_devices": [3]
//    }
}