from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset
import torch
from sklearn.metrics import accuracy_score,classification_report
from tqdm import tqdm
import argparse
from utils import create_optimizer
from transformers.trainer_utils import SchedulerType
import math
from transformers.optimization import get_scheduler
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import logging
from transformers import set_seed
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
sys.path.append("./")
class Collator():
    def __init__(self,max_seq_length,tokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __call__(self, examples):
        res_examples = []
        text = [ example["text"] for example in examples ]
        labels =[ example["label"] for example in examples ]
        proc_data = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            padding="longest",
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        labels = torch.tensor(labels)

        return proc_data,labels

@torch.no_grad()
def test(test_loader,model,device):
    model.eval()
    y_true,y_pred=[],[]
    for batch in tqdm(test_loader):
        input_data,labels = batch
        # labels.to(device)
        for k in input_data:
            input_data[k]=input_data[k].to(device)

        output = model(**input_data)
        predict = output["logits"].argmax(dim=1)
        predict = predict.cpu().tolist()
        labels = labels.cpu().tolist()
        y_pred.extend(predict)
        y_true.extend(labels)
    model.train()
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    return acc

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    set_seed(args.seed)
    collator=Collator(max_seq_length=args.max_length, tokenizer=tokenizer)
    dataset = load_dataset("json",data_files=args.train_path)
    train_loader = DataLoader(
            dataset["train"],
            batch_size=args.batch_size,
            collate_fn=collator,
            shuffle=True,
            drop_last=False,
            num_workers=10,
            pin_memory=True,
        )
    ours_dataset = load_dataset("json",data_files={"validation":args.hc3_si_val_path})
    ours_val_loader = DataLoader(
            ours_dataset["validation"],
            batch_size=args.batch_size,
            collate_fn=collator,
            shuffle=True,
            drop_last=False,
            num_workers=10,
            pin_memory=True,
        )
    hc3_dataset = load_dataset("json",data_files={"validation":args.hc3_val_path})
    hc3_val_loader = DataLoader(
            hc3_dataset["validation"],
            batch_size=args.batch_size,
            collate_fn=collator,
            shuffle=True,
            drop_last=False,
            num_workers=10,
            pin_memory=True,
        )
    optimizer = create_optimizer(args,model)
    os.makedirs(args.tensorboard_dir,exist_ok=True)
    os.makedirs(args.save_path,exist_ok=True)
    writer = SummaryWriter(args.tensorboard_dir)

    num_training_steps_per_epoch = math.ceil(len(train_loader.dataset) // args.batch_size)
    num_training_steps = num_training_steps_per_epoch * args.epochs
    num_warmup_steps = num_training_steps * args.warm_up_ratio
    test_steps = num_training_steps // args.num_test_times

    global_step = 0
    best_ours_acc = -1
    best_hc3_acc = -1
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader):
            input_data,labels = batch
            labels=labels.to(device)
            for k in input_data:
                input_data[k]=input_data[k].to(device)
            output = model(**input_data,labels=labels)
            optimizer.zero_grad()
            loss = output.loss
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('loss', loss.item(), global_step)
            # writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
            
            if (global_step+1) % test_steps==0:
                ours_val_acc = test(ours_val_loader,model,device)
                hc3_val_acc = test(hc3_val_loader,model,device)
                writer.add_scalar('ours_val_acc', ours_val_acc, global_step)
                writer.add_scalar('hc3_val_acc', hc3_val_acc, global_step)
                logger.info(f"ours acc:{ours_val_acc}")
                logger.info(f"hc3 acc:{hc3_val_acc}")

                if ours_val_acc >= best_ours_acc:
                    best_ours_acc = ours_val_acc
                    save_path = args.save_path + os.sep + "ours"
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logger.info(f"best ours acc:{best_ours_acc}")
                if hc3_val_acc >= best_hc3_acc:
                    best_hc3_acc = hc3_val_acc
                    save_path = args.save_path + os.sep + "hc3"
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logger.info(f"best hc3 acc:{hc3_val_acc}")

            global_step+=1
            # scheduler.step()
        
def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--train_path',  type=str)
    parser.add_argument('--hc3_val_path',  type=str)
    parser.add_argument('--hc3_si_val_path', type=str)
    parser.add_argument('--model', default="roberta-base", type=str)
    parser.add_argument('--max_length', default=64,type=int)
    parser.add_argument('--batch_size',default=4,type=int)
    parser.add_argument('--save_path',type=str,help="save folder")
    parser.add_argument('--tensorboard_dir', type=str,help="save folder")
    parser.add_argument('--lr',default=5e-5,type=float)
    parser.add_argument('--weight_decay',default=0,type=float)
    parser.add_argument('--warm_up_ratio',default=0.0,type=float)
    parser.add_argument('--epochs',default=2,type=int)
    parser.add_argument('--num_test_times', default=10, type=int,
                        help='number of verifications')
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parser_args() 
    main(args)