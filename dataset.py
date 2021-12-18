import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path,block_size,bos='[BOS]',eos='[EOS]'):
        self.data = []
        self.file_path = file_path
        self.tokenizer = tokenizer

        self.examples = []
        text = ""

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = bos+line+eos 
                text += line   
        tokenized_text = self.tokenizer.encode(text)


        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  
            self.examples.append(
                tokenized_text[i : i + block_size]
            )
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
