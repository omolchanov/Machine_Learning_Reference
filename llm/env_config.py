DATA_DIRECTORY = 'data'

# === Tokenizer config ===
TOK_FILENAME = 'tokenizer.pkl'
TOK_PATHNAME = f"{DATA_DIRECTORY}/{TOK_FILENAME}"
TOK_OBJ_FILENAME = 'tokenizer_obj.pkl'
TOK_OBJ_PATHNAME = f"{DATA_DIRECTORY}/{TOK_OBJ_FILENAME}"


# === Dataset config ===
DS_FIlENAME = 'dataset.npy'
DS_MD_FILENAME = 'ds_metadata.json'
DS_PATHNAME = f"{DATA_DIRECTORY}/{DS_FIlENAME}"
DS_MD_PATHNAME = f"{DATA_DIRECTORY}/{DS_MD_FILENAME}"

DATASET_SIZE = 10

# === Model's config ===
MODELS_DIRECTORY_PATH = 'models'
MODEL_METADATA_FILENAME = 'model_metadata.json'
CHECKPOINTS_DIRECTORY = 'checkpoints'

# === Model's hyperparameters
BLOCK_SIZE = 128
BATCH_SIZE = 64
EMBED_DIMS = 64
NUM_HEADS = 2
FF_DIM = 128
EPOCHS = 1

# === Chat config
ANSWER_LENGTH = 20
