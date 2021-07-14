import torch

# Load PhoBERT-base in fairseq
from fairseq.models.roberta import RobertaModel
phobert = RobertaModel.from_pretrained('../PhoBERT_base_fairseq', checkpoint_file='model.pt')
phobert.eval()  # disable dropout (or leave in train mode to finetune)

# Incorporate the BPE encoder into PhoBERT-base 
from fairseq.data.encoders.fastbpe import fastBPE  
from fairseq import options  
parser = options.get_preprocessing_parser()  
parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE', default="../PhoBERT_base_fairseq/bpe.codes")  
args = parser.parse_args()  
phobert.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT

# INPUT TEXT IS WORD-SEGMENTED!
line = "Tao là sinh_viên trường đại_học Công_nghệ ."  

# Extract the last layer's features
tokenized = phobert
subwords = phobert.encode(line)  
print(args)
# last_layer_features = phobert.extract_features(subwords)  
# assert last_layer_features.size() == torch.Size([1, 9, 768])  
  
# # Extract all layer's features (layer 0 is the embedding layer)  
# all_layers = phobert.extract_features(subwords, return_all_hiddens=True)  
# assert len(all_layers) == 13  
# assert torch.all(all_layers[-1] == last_layer_features)  

# # Filling marks  
# masked_line = 'Tôi là  <mask> trường đại_học Công_nghệ .'  
# topk_filled_outputs = phobert.fill_mask(masked_line, topk=5)  
# print(topk_filled_outputs)