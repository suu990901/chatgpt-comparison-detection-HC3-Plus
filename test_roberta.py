from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset
import torch
from sklearn.metrics import accuracy_score,classification_report
from tqdm import tqdm
from run_roberta import Collator
from run_roberta import test

def main(data_path,model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset("json",data_files=data_path,split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    collator=Collator(max_seq_length=256, tokenizer=tokenizer)
    test_loader = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=collator,
            shuffle=True,
            drop_last=False,
            num_workers=10,
            pin_memory=True,
        )

    acc=test(test_loader,model,device)
    print(acc)

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path',  type=str)
    return parser.parse_args()

if __name__=="__main__":
    args = parser_args() 
    main(args.data_path,args.model_path)

