from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import numpy as np
from flask import Flask, request, jsonify
from transformers import pipeline
from math import ceil
from langchain.text_splitter import RecursiveCharacterTextSplitter
tokenizer = None
max_context = 512
import os


model_choice = int(input(
"""
0. CohereForAI/aya-101
1. meditron
2. falconsai/medical_sumarization
3. summa
4. meta-llama/Llama-2-70b-chat-hf
5. meta-llama/Llama-2-13b-chat-hf
6. meta-llama/Llama-2-7b-chat-hf
7. mistralai/Mixtral-8x7B-Instruct-v0.1
8. google/gemma-7b-it
9. openai/gpt-3.5
10. sweditron-13b
11. sweditron-13b-ft
12. sweditron-13b-ft-lora434
13. meta-llama/Meta-Llama-3-8B-Instruct
"""))
#----------
if model_choice == 0:
    from transformers import AutoModelForSeq2SeqLM
    max_tokens = 4096
    model_name = "CohereForAI/aya-101"
    tokenizer = AutoTokenizer.from_pretrained(model_name,token='')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                 device_map='auto',
                                                 torch_dtype=torch.float32,
                                                 local_files_only=False)
    max_context = 4096
elif model_choice == 1:
    max_tokens = 4000
    model_name = "malhajar/meditron-70b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name,token='')
    model = AutoModelForCausalLM.from_pretrained('/workspace/data/'+model_name,
                                                 device_map='auto',
                                                 torch_dtype=torch.float16,
                                                 local_files_only=True)
    max_context = model.config.max_position_embeddings
elif model_choice == 2:
    model_name="Falconsai/medical_summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name,token='')
    summarizer = pipeline("summarization", model="Falconsai/medical_summarization",device_map = 'auto')
#----------
elif model_choice == 3:
    from summa import summarizer,keywords
    
elif model_choice == 4:
    max_tokens = 4000
    model_name = "meta-llama/Llama-2-70b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=' ')
    model = AutoModelForCausalLM.from_pretrained(model_name,token=' ',
                                                 device_map='auto',
                                                 torch_dtype=torch.float16)
    max_context = model.config.max_position_embeddings
    
elif model_choice == 5:
    max_tokens = 4000
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=' ')
    model = AutoModelForCausalLM.from_pretrained(model_name,token=' ',
                                                 device_map='auto',
                                                 torch_dtype=torch.float32)
    max_context = model.config.max_position_embeddings
    
elif model_choice == 6:
    max_tokens = 4000
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=' ')
    model = AutoModelForCausalLM.from_pretrained(model_name,token=' ',
                                                 device_map='auto',
                                                 torch_dtype=torch.float16)
    max_context = model.config.max_position_embeddings
    
elif model_choice == 7:
    max_tokens = 30000
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=' ')
    model = AutoModelForCausalLM.from_pretrained(model_name,token=' ',
                                                 device_map='auto',
                                                 torch_dtype=torch.bfloat16)
    max_context = model.config.max_position_embeddings
    
elif model_choice == 8:
    max_tokens = 8000
    model_name = "google/gemma-7b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=' ')
    model = AutoModelForCausalLM.from_pretrained(model_name,token=' ',
                                                 device_map='auto',
                                                 torch_dtype=torch.bfloat16)
    max_context = model.config.max_position_embeddings

elif model_choice == 9:
    from openai import AzureOpenAI
    client = client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    model_name = "gpt-35-turbo-16k"
    
elif model_choice == 10:
    max_tokens = 4000
    model_name = '../../data/pretraining/qa_nodomain'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_name = model_name
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map='auto',
                                                 torch_dtype=torch.float32,local_files_only = True)
    max_context = model.config.max_position_embeddings
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    
elif model_choice == 11:
    from peft import PeftModel
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    max_tokens = 4000
    tokenizer = AutoTokenizer.from_pretrained('../../data/finetuned/lora')
    m = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.float32,
                                                 device_map='auto',token=' ')
    m = PeftModel.from_pretrained(m,'../../data/finetuned/lora')
    #print('get_nb_trainable_parameters ',m.get_nb_trainable_parameters())
    print('print_trainable_parameters ',m.print_trainable_parameters())
    model = m.merge_and_unload(safe_merge=True)
    max_context = model.config.max_position_embeddings

elif model_choice == 12:
    from peft import PeftModel
    model_name = '../../data/pretraining/qa_nodomain'
    max_tokens = 4000
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=' ')
    m = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.float32,
                                             device_map='auto',
                                             token=' '
                                            )
    
    m = PeftModel.from_pretrained(m,'../../data/finetuned/lora_mega_no_domain')
    model = m.merge_and_unload()
    max_context = model.config.max_position_embeddings
    
elif model_choice == 13:
    max_tokens = 4000
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=' ')
    model = AutoModelForCausalLM.from_pretrained(model_name,token=' ',
                                                 device_map='auto',
                                                 torch_dtype=torch.bfloat16)
    max_context = model.config.max_position_embeddings
    
if tokenizer is not None and tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


app = Flask(__name__)

@app.route('/set_cfg',methods=["POST"])
def set_cfg():
    try:
        cfg = GenerationConfig.from_pretrained(model_name,
                                               token=' ').to_dict()
        data = request.get_json()
        #print(type(data),data)
        #print(dir(model.generation_config))

        for key in data.keys():
            if data[key] <= 0:
                continue
            cfg[key] = data[key]
            if key == "do_sample":
                del cfg["top_p"]
                del cfg["temperature"]
            
        model.generation_config = GenerationConfig.from_dict(cfg)
        print("FINAL FANTASY", model.generation_config)
        return jsonify({"response":":rocket:"}),200
    except Exception as e:
        print(str(e))
        return jsonify({"error":str(e)}),500
        


@app.route('/llama_tokenize', methods=['POST'])
def process_llama_tokenize():
   # Get the input string from the request
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    if 'prompt_template' not in data:
        return jsonify({'error': 'No prompt template provided'}), 400
    if 'prompt_reduce' not in data:
        return jsonify({'error': 'No prompt reducer provided'}), 400

    text = data['text']
    prompt_template = data['prompt_template']
    prompt_reduce = data['prompt_reduce']
    prompt_template_len = len(prompt_template)
    prompt_template_tokenized_len = len(tokenizer.encode(prompt_template, return_tensors="pt")[0])
    index = prompt_template.find('*')
    index_reduce = prompt_reduce.find('*')
    
    try:
        #print("Initial text: \n", text)
        inputs = tokenizer.encode(text, return_tensors="pt")#.to('cuda')
        text_len = len(inputs[0])
        while text_len + prompt_template_tokenized_len > max_context:
            splitted = split_the_text(text=text,max_chunk_size=ceil(max_context/2))
            print(len(splitted))
            #print(chunk_length)
            text = ""
            for i, split in enumerate(splitted):
                prompt = prompt_template[:index-1] + split + prompt_template[index+1:]
                inputs = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
                outputs = model.generate(
                    inputs,max_length = max_tokens,
                    pad_token_id=tokenizer.pad_token_id)
                result = tokenizer.decode(outputs[0])[len(prompt)+4:-4]
                text += f"{i}. {result}"
            inputs = tokenizer.encode(text, return_tensors="pt")
            text_len = len(inputs[0])
        
        prompt = prompt_reduce[:index_reduce-1] + text + prompt_reduce[index_reduce+1:]
        inputs = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(
                    inputs,max_length = max_tokens,
                    pad_token_id=tokenizer.pad_token_id)
        result = tokenizer.decode(outputs[0])[len(prompt)+4:-4]
        del inputs
        #print(result)
        return jsonify({'response': result}), 200
    except Exception as e:
        print(len(inputs[0]))
        return jsonify(
            {'error':str(e),
             'input_len':text_len,
             'prompt_len':prompt_template_tokenized_len,
             'prompt':prompt,'text':text,
             'reduce_index':index_reduce}), 500

    
@app.route('/dspy_optimize', methods=['POST'])
def process_dspy_optimize():
   # Get the input string from the request
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    try:
        inputs = tokenizer.encode(text, return_tensors="pt").to('cuda')
        outputs = model.generate(
                    inputs,max_length = max_tokens,
                    pad_token_id=tokenizer.pad_token_id)
        result = tokenizer.decode(outputs[0])[len(text)+4:-4]
        del inputs
        #print(result)
        return jsonify({'response': result}), 200
    except Exception as e:
        print(len(inputs[0]))
        print(str(e))
        return jsonify(
            {'error':str(e)}), 500
    
@app.route('/falcons_summarize', methods=['POST'])
def falcons_summarizer():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    try:
        while len(tokenizer.encode(text, return_tensors="pt")[0]) > 512:
            chunks = split_the_text(text=text,max_chunk_size=256)
            summaries = [summarizer(chunk, max_length=128, do_sample=False)[0]['summary_text'] for chunk in chunks]
            text = "\n\n".join(summaries)
        text = summarizer(text, max_length=400, do_sample=False)[0]['summary_text']
        return jsonify({"response":text}),200
    except Exception as e:
        return jsonify({"error":str(e)}),500
        
@app.route('/summa', methods=['POST'])
def summa_summarizer():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    try:
        result = summarizer.summarize(text) + '\n'
        #result += keywords.keywords(text)
        return jsonify({"response":result}),200
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route('/openai', methods=['POST'])
def openai_summarizer():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    try:
        messages = [
            {
                "role":"system",
                "content":"Du är en hjälpsam medicinsk assistent som hjälper läkare och sjuksköterskor genom att sammanfatta information om patienter. Svara med en punktlista för varje kategori i den givna mallen. Svara alltid på svenska."
            },
            {
                "role":"user",
                "content":f"Nedan ges anamnes för en patient under en dag\n<anamnes>\n{text}\n</anamnes>\nDu ska plocka ut information som passar i mallen nedan. Undvik onödig information och plocka endast ut sådant som rör varje rubrik. Om relevant information saknas så lämnar du rubriken tom.\nFormattera ditt svar enligt mallen.\n<mall>\n*Sjukdomshistoria (Patientens diagnoser, sjukdomshistorik och riskfaktorer (t.ex. sjukdomar i familjen))*\n\n*Sökorsaker (Patientens symtom och/eller datum för ingrepp)*\n\n*Åtgärder (Planerade undersökningar, behandlingar och åtgärder)*\n\n</mall>"
            }
        ]

        response = client.chat.completions.create(
          model=model_name,
          #response_format={ "type": "json_object" },
          messages=messages
        )
        result = response.choices[0].message.content
        return jsonify({"response":result}),200
    except Exception as e:
        return jsonify({"error":str(e)}),500
    
def split_the_text(text : str, max_chunk_size : int) -> [str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=50,
        length_function= lambda x : len(tokenizer.encode(x,  return_tensors="pt")[0]),
        is_separator_regex=False,
    )
    splitted = text_splitter.split_text(text)
    return splitted
    
if __name__ == '__main__':
    app.run(host='localhost', port=8000)