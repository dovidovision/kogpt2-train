from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast, \
                         Trainer, TrainingArguments, GPT2LMHeadModel
import torch
from dataset import TextDataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Find tuning ko-gpt2")
    parser.add_argument('--data-path',type=str,default='./data/cat_diary_data.txt',help='Dataset path')
    parser.add_argument('--save-dir',type=str,default='work_dirs/model_output',help='path that model is saved.')
    parser.add_argument('--epochs',type=int,default=30,help='training epochs')
    parser.add_argument('--batch-size',type=int,default=32,help='training batch size')
    parser.add_argument('--block-size',type=int,default=64,help='Dataset block size')

    args = parser.parse_args()
    return args

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
    bos_token='<s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>') 

    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2').to(device)

    data = TextDataset(tokenizer,file_path=args.data_path,block_size=args.block_size,bos='<s>',eos='</s>')

    data_collator = DataCollatorForLanguageModeling( 
        tokenizer=tokenizer, mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, # 512:32  # 128:64
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=data
    )

    trainer.train()
    trainer.save_model()

if __name__=='__main__':
    args = parse_args()
    main(args)
