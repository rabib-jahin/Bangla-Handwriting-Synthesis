import torch

###############################################

EXP_NAME = "Final_Bangla"; RESUME = True

DATASET = 'BN'
if DATASET == 'BN':
    DATASET_PATHS = 'BN-UNIFIED-NO-SINGLE.pickle'
    NUM_WRITERS = 817  
if DATASET == 'CVL':
    DATASET_PATHS = 'files/CVL-32.pickle'
    NUM_WRITERS = 283
ENGLISH_WORDS_PATH = 'BN-EN-WORDS.txt'

###############################################

IMG_HEIGHT = 32
resolution = 16
batch_size = 8
NUM_EXAMPLES = 15#15
TN_HIDDEN_DIM = 512
TN_DROPOUT = 0.1
TN_NHEADS = 8
TN_DIM_FEEDFORWARD = 512
TN_ENC_LAYERS = 3
TN_DEC_LAYERS = 3
ch='"'
ALPHABET = " !#%&'()*+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\^abcdefghijklmnopqrstuvwxyz|θπ।ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎৗড়ঢ়য়০১২৩৪৫৬৭৮৯৷‌‍–—"
ALPHABET=ch+ALPHABET
# for char in BN_ALPHABET:
#     if char not in ALPHABET: ALPHABET += char

VOCAB_SIZE = len(ALPHABET)
G_LR = 0.00005
D_LR = 0.00005
W_LR = 0.00005
OCR_LR = 0.00005
EPOCHS = 1000
NUM_CRITIC_GOCR_TRAIN = 2
NUM_CRITIC_DOCR_TRAIN = 1
NUM_CRITIC_GWL_TRAIN = 2
NUM_CRITIC_DWL_TRAIN = 1
NUM_FID_FREQ = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_SEQ = True
NUM_WORDS = 3
if not IS_SEQ: NUM_WORDS = NUM_EXAMPLES
IS_CYCLE = False
IS_KLD = False
ADD_NOISE = False
ALL_CHARS = False
SAVE_MODEL = 5
SAVE_MODEL_HISTORY = 100

def init_project():
    import os, shutil
    if not os.path.isdir('saved_images'): os.mkdir('saved_images')
    if os.path.isdir(os.path.join('saved_images', EXP_NAME)): shutil.rmtree(os.path.join('saved_images', EXP_NAME))
    os.mkdir(os.path.join('saved_images', EXP_NAME))
    os.mkdir(os.path.join('saved_images', EXP_NAME, 'Real'))
    os.mkdir(os.path.join('saved_images', EXP_NAME, 'Fake'))

