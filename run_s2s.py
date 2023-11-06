import sys
# sys.path.append("../")
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer,AutoConfig
)
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
from torch.utils.tensorboard import SummaryWriter
import logging
from transformers import set_seed
from instruction import InstructionsHandler_Chinese,InstructionsHandler_English
# from utils import OursGenerator
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class trainCollator():
    def __init__(self,max_seq_length,tokenizer,language):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.map = {"0":"human","1":"model"}
        self.seq2seq_collator = DataCollatorForSeq2Seq(self.tokenizer)
        if language=="en":
            self.instructhandler = InstructionsHandler_English()
        elif language=="zh":
            self.instructhandler = InstructionsHandler_Chinese()
        else:
            raise ValueError("LANGUAGE ERROR")

    def __call__(self, examples):
        t_examples = [ example["text"] for example in examples ]
        labels = [ example["label"] for example in examples ]
        text = []
        for example in t_examples:
            instruct = self.instructhandler.load_instruction_set()
            t=instruct['input_instruct']+example+instruct['eos_instruct']
            text.append(t)
        text_labels = []
        for label in labels:
            text_labels.append(self.map[str(label)])

        all_data = []
        for i in range(len(text)):
            row_text,row_label = text[i],text_labels[i]
            proc_data = self.tokenizer(
                row_text,
                truncation=True,
                max_length=self.max_seq_length,
                # return_tensors="pt"
            )
            proc_label = self.tokenizer(
                row_label,
                truncation=True,
                max_length=self.max_seq_length,
                # return_tensors="pt"
            )
            proc_data["labels"] = proc_label["input_ids"]
            all_data.append(proc_data)
        all_data = self.seq2seq_collator(all_data)
        return all_data

class evalCollator():
    def __init__(self,max_seq_length,tokenizer,language):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.seq2seq_collator = DataCollatorForSeq2Seq(self.tokenizer)
        if language=="en":
            self.instructhandler = InstructionsHandler_English()
        elif language=="zh":
            self.instructhandler = InstructionsHandler_Chinese()
        else:
            raise ValueError("LANGUAGE ERROR")

    def __call__(self, examples):

        t_examples = [ example["text"] for example in examples ]
        text = []
        for example in t_examples:
            instruct = self.instructhandler.load_instruction_set()
            t=instruct['input_instruct']+example+instruct['eos_instruct']
            text.append(t)
        labels =[ example["label"] for example in examples ]

        proc_data = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="longest",
            return_tensors="pt"
        )
        proc_data["labels"] = labels
        return proc_data

@torch.no_grad()
def test(test_loader,model,tokenizer,device):
    model.eval()
    y_true,y_pred=[],[]
    for batch in tqdm(test_loader):
        input_ids,labels = batch["input_ids"].to(device),batch["labels"]
        output = model.generate(input_ids)
        output_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
        for output_text in output_texts:
            if output_text.lower() == "human":
                predict = 0
            elif output_text.lower() == "model":
                predict = 1
            else:
                predict = 2
            y_pred.append(predict)
        y_true.extend(labels)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    model.train()
    return acc

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)
    set_seed(args.seed)
    train_collator=trainCollator(max_seq_length=args.max_length, tokenizer=tokenizer,language=args.lang)
    eval_collator=evalCollator(max_seq_length=args.max_length, tokenizer=tokenizer,language=args.lang)
    dataset = load_dataset("json",data_files=args.train_path)
    train_loader = DataLoader(
            dataset["train"],
            batch_size=args.batch_size,
            collate_fn=train_collator,
            shuffle=True,
            drop_last=False,
            num_workers=10,
            pin_memory=True,
        )
    ours_dataset = load_dataset("json",data_files={"validation":args.hc3_si_val_path})
    ours_val_loader = DataLoader(
            ours_dataset["validation"],
            batch_size=args.batch_size,
            collate_fn=eval_collator,
            shuffle=True,
            drop_last=False,
            num_workers=10,
            pin_memory=True,
        )
    hc3_dataset = load_dataset("json",data_files={"validation":args.hc3_val_path})
    hc3_val_loader = DataLoader(
            hc3_dataset["validation"],
            batch_size=args.batch_size,
            collate_fn=eval_collator,
            shuffle=True,
            drop_last=False,
            num_workers=10,
            pin_memory=True,
        )
    optimizer = create_optimizer(args,model)
    os.makedirs(args.tensorboard_dir,exist_ok=True)
    os.makedirs(args.save_path,exist_ok=True)
    writer = SummaryWriter(args.tensorboard_dir)

    effect_batch = args.batch_size * args.accumulation_steps
    num_training_steps_per_epoch = math.ceil(len(train_loader.dataset) // effect_batch)
    num_training_steps = num_training_steps_per_epoch * args.epochs
    num_warmup_steps = num_training_steps * args.warm_up_ratio
    test_steps = num_training_steps // args.num_test_times
    scheduler = get_scheduler(
                SchedulerType.LINEAR,
                # SchedulerType.CONSTANT,
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

    global_step = 0
    best_ours_acc = -1
    best_hc3_acc = -1
    accumulation_steps = args.accumulation_steps
    for epoch in range(args.epochs):
        model.train()
        for model_inputs in tqdm(train_loader):
            if global_step%accumulation_steps==0:
                scheduler.step()
    
            for k in model_inputs:
                model_inputs[k]=model_inputs[k].to(device)
            output = model(**model_inputs)
            loss = output.loss
        
            acc_loss = loss/accumulation_steps
            acc_loss.backward()
            if((global_step+1)%accumulation_steps)==0:
                optimizer.step()
                optimizer.zero_grad() 

            writer.add_scalar('loss', loss.item(), global_step)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
            
            if (global_step+1) % test_steps==0:
            # if global_step % 1==0:
                ours_val_acc = test(ours_val_loader,model,tokenizer,device)
                hc3_val_acc = test(hc3_val_loader,model,tokenizer,device)
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
        
def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--train_path',  type=str)
    parser.add_argument('--hc3_val_path',type=str)
    parser.add_argument('--hc3_si_val_path', type=str)
    parser.add_argument('--model', default="chinese_tk_base", type=str)
    parser.add_argument('--max_length', default=512,type=int)
    parser.add_argument('--batch_size',default=4,type=int)
    parser.add_argument('--save_path', type=str,help="save folder")
    parser.add_argument('--tensorboard_dir', type=str,help="save folder")
    parser.add_argument('--lr',default=5e-5,type=float)
    parser.add_argument('--weight_decay',default=0,type=float)
    parser.add_argument('--warm_up_ratio',default=0.0,type=float)
    parser.add_argument('--epochs',default=10,type=int)
    parser.add_argument('--num_test_times', default=1, type=int,
                        help='number of verifications')
    parser.add_argument('--lang', default="zh", type=str,
                        help='language')
    parser.add_argument('--accumulation_steps', default=2, type=int)                        
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parser_args() 
    main(args)