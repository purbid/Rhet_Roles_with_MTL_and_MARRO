# Rhet_Roles_with_MTL_and_MARRO
For rhetorical role classification from legal texts using a transformer-LSTM-CRF, trained in a multi task learning setup


for TF-BiLSTM-CRF (baseline model)

on India dataset

python [run.py](http://run.py/) --use_tf_lstm_crf True —data_path `/dataset/pretrained_embeddings/IN-train-set/` —`dataset_size` 150

on UK dataset

python [run.py](http://run.py/) --use_tf_lstm_crf True —data_path `/dataset/pretrained_embeddings/UK-train-set/` —`dataset_size` 50

For OUR models :

TF-MARRO:

on India Dataset:
python run.py --use_tf_emb_plus_attention True  --dataset_size 50 --attention_heads 8  —data_path `/dataset/IN-train-set/` —`dataset_size` 150 `--batch_size 1`

on UK Dataset:
python run.py --use_tf_emb_plus_attention True  --dataset_size 50 --attention_heads 8  —data_path `/dataset/UK-train-set/`  `--batch_size 1`
