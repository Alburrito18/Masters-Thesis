from transformers import AutoModelForCausalLM, TrainingArguments,GenerationConfig, AutoTokenizer
from peft import LoraModel, LoraConfig
import pandas as pd
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from sklearn.model_selection import train_test_split
import datetime
import torch
import evaluate
from datasets import Dataset
import wandb

def compute_metrics(test,model,tokenizer):
    max_tokens = 2000
    rouge = evaluate.load('rouge')
    bleu = evaluate.load("bleu")
    meteor = evaluate.load('meteor')
    answers = list()
    references = list()
    
    for i,instance in enumerate(test):
        prompt = instance['prompt']
        inputs = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(
                    inputs,max_length = max_tokens,
                    pad_token_id=tokenizer.pad_token_id)
        result = tokenizer.decode(outputs[0])[len(prompt)+3:-4]
        answers.append(result)
        references.append(instance['completion'])
        if i % 100 == 0:
            with open("results.txt", "a") as file:
                file.write(f"{str(i)} : {str(datetime.datetime.now())}")
            print(i,datetime.datetime.now())
    bleu_score = bleu.compute(predictions = answers, references = references)
    rouge_score = rouge.compute(predictions = answers, references = references)
    meteor_score = meteor.compute(predictions = answers, references = references)
    print(bleu_score)
    print(rouge_score)
    print(meteor_score)
    with open("results.txt","a") as file:
        file.write(f"bleu:{str(bleu_score)}\nrouge:{str(rouge_score)}\nmeteor:{str(meteor_score)}\n\n")

def make_cfg():
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.01,
        task_type="CAUSAL_LM"
    )
    tokenizer = AutoTokenizer.from_pretrained("../../../data/artifacts/checkpoint-cme0pvg4:v2")
    model = AutoModelForCausalLM.from_pretrained("../../../data/artifacts/checkpoint-cme0pvg4:v2",
                                                torch_dtype=torch.float32,
                                                 device_map='auto')
    cfg = {'renormalize_logits': True, 'typical_p': 0.5918630300249034, 'temperature': 1.6680717661016695, 'do_sample': True, 'epsilon_cutoff': 0.0006978287432014012}
    model.generation_config = GenerationConfig.from_dict(cfg)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    
    lora_model = LoraModel(model, config, "default")
    return lora_model, tokenizer

                                                 
def make_trainer(train, test, model, tokenizer):
    args = TrainingArguments(
        output_dir="lora_ft_March_13",
        #per_device_train_batch_size=1,
        auto_find_batch_size=True,
        #per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=900,
        logging_steps=1,
        #gradient_accumulation_steps=10,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=3_00,
        lr_scheduler_type="cosine",
        learning_rate=2e-5,
        save_steps=18_000,
        fp16=False,
        push_to_hub=False,
        report_to="wandb",
        adam_beta2 = 0.95,
        adam_epsilon = 1e-5,
        neftune_noise_alpha=5,
        remove_unused_columns=False
    )
    response_template = "\n[/INST]"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer,mlm=False)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        #data_collator=collator,
        train_dataset=train,
        eval_dataset=test,
        packing=True,
        max_seq_length=2048,
        dataset_text_field='text'
    )
    return trainer

prompt1 = '''[INST] <<SYS>>
Du är en hjälpsam medicinsk assistent som hjälper läkare och sjuksköterskor genom att svara på frågor.
Svara på svenska.
<</SYS>>
Nedan ges en fråga eller ett medicinskt begrepp.
<fråga>'''

prompt2 = '''
</fråga>
Svara på frågan eller förklara begreppet.
[/INST]'''
data = pd.read_csv("test.csv")

formatted = []
for index, row in data.iterrows():
    #print(row)
    elem = {'text':f"<s>{prompt1} {row['Question']} {prompt2} {row['Answer']} </s>"}
    formatted.append(elem)
    

train, test = train_test_split(formatted,test_size = 0.05,random_state = 42)
train = Dataset.from_list(train)
test = Dataset.from_list(test)

model, tokenizer = make_cfg()
#compute_metrics(test,model,tokenizer)

trainer = make_trainer(train,test,model, tokenizer)
trainer.train()
trainer.evaluate()
trainer.save_model('../../../data/finetuned/')
wandb.finish()