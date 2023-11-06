from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
)
from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset
import torch
from sklearn.metrics import accuracy_score,classification_report
from tqdm import tqdm
from run_s2s import test
from run_s2s import evalCollator
from instruction import InstructionsHandler_Chinese,InstructionsHandler_English

class mul_instruction_evalCollator():
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
        l =[ example["label"] for example in examples ]
        text = []
        labels = []
        for index,example in enumerate(t_examples):
            instructions = self.instructhandler.all_instruct
            for instruction in instructions:
                t=instruction+example+self.instructhandler.eos
                text.append(t)
                labels.append(l[index])

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
def mul_test(test_loader,model,tokenizer,device,language):
    if language=="en":
        instructhandler = InstructionsHandler_English()
    elif language=="zh":
        instructhandler = InstructionsHandler_Chinese()
    else:
        raise ValueError("LANGUAGE ERROR")

    instructions_number = len(instructhandler.all_instruct)
    model.eval()
    y_true,y_pred=[],[]
    for batch in tqdm(test_loader):
        input_ids,labels = batch["input_ids"].to(device),batch["labels"]
        output = model.generate(input_ids)
        output_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
        # predict = [ for output_text in output_texts ]
        for output_text in output_texts:
            if output_text.lower() == "human":
                predict = 0
            elif output_text.lower() == "model":
                predict = 1
            else:
                predict = 2
            y_pred.append(predict)
        y_true.extend(labels)
    
    proc_y_pred,proc_y_true=[],[]
    number_zero,number_one = 0,0
    for index,y in enumerate(y_pred):
        if y==0 :
            number_zero+=1
        elif y==1:
            number_one+=1
        if (index+1)%instructions_number==0:
            true_label = y_true[index]
            proc_y_true.append(true_label)
            predict_label =0 if number_zero>number_one else 1
            proc_y_pred.append(predict_label)
            number_one,number_zero=0,0
    acc = accuracy_score(y_true=proc_y_true, y_pred=proc_y_pred)
    model.train()
    return acc

def main(data_path,model_path,lang,mul=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset("json",data_files=data_path,split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    if mul==False:
        collator=evalCollator(max_seq_length=512, tokenizer=tokenizer,language=lang)
    else: 
        collator=mul_instruction_evalCollator(max_seq_length=512, tokenizer=tokenizer,language=lang)
    test_loader = DataLoader(
            dataset,
            batch_size=40,
            collate_fn=collator,
            shuffle=False,
            drop_last=False,
            num_workers=10,
            pin_memory=True,
        )
    if mul==False:
        acc = test(test_loader,model,tokenizer,device)
    else:
        acc = mul_test(test_loader,model,tokenizer,device,lang)
    print(acc)

if __name__=="__main__":
    def parser_args():
        parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--lang', default="en", type=str, help="The language of the task being tested", choices=["en","zh"])
    return parser.parse_args()

if __name__=="__main__":
    args = parser_args() 
    main(args.data_path,args.model_path,args.lang)
