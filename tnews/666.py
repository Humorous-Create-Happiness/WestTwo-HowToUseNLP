from torch import LongTensor
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert = BertModel.from_pretrained("bert-base-chinese")
onehot_list = tokenizer.encode("某[MASK]文字", padding="max_length", max_length = 64)

bert.eval()
input_tensor = LongTensor([onehot_list])
output_tensor = bert(input_tensor)[0]