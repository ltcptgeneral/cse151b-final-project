
from torch.utils.data import Dataset
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer
from tqdm import tqdm as progress_bar
import torch
import matplotlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

encoder = BertGenerationEncoder.from_pretrained("google-bert/bert-large-uncased", bos_token_id=101, eos_token_id=102)
# add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
decoder = BertGenerationDecoder.from_pretrained("google-bert/bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

# create tokenizer...
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")

import json

class CodeDataset(Dataset):
    def __init__(self):
        with open("data/conala-train.json") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        intent = self.data[idx]["rewritten_intent"] if self.data[idx]["rewritten_intent"] else self.data[idx]["intent"]
        return intent, self.data[idx]["snippet"]


optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
dataloader = CodeDataset()
model = model.to(device)

losses = []
epochs = 10
for i in range(epochs):

    epoch_loss = 0

    for idx, (question, answer) in progress_bar(enumerate(dataloader), total=len(dataloader)):

        print(question)
        input_ids = tokenizer(question, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        label_ids = tokenizer(answer, return_tensors="pt").input_ids.to(device)

        loss = model(input_ids=input_ids, decoder_input_ids=label_ids, labels=label_ids).loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    losses.append(epoch_loss)

plt.plot(losses, color="green", label="Training Loss")
plt.legend(loc = 'upper left')
plt.show()


