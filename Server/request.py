import requests
import json


def call_api(prompt, options, context):
    prompt_reduce = options['config']['promptReducer']
    url = 'http://localhost:8000'+ options['config']['additionalOption']#['additionalOption']#.get('additionalOption', None)
    prompt_template = options['config']['promptTemplate'] 
    with open(prompt_reduce,"r") as f:
        prompt_reducer = " ".join(f.readlines())
        
    with open(prompt_template,"r") as f:
        prompt_template = " ".join(f.readlines())
    
    data = {
            "text":prompt, 
            "prompt_template":prompt_template,
            "prompt_reduce":prompt_reducer
           }

    headers = {"Content-Type":"application/json"}
    response = requests.post(url,headers=headers, data=json.dumps(data))

    result = {"output":'temp',}
    
    if response.status_code == 200:
        #print('Response successful!')
        result['output'] = response.json()["response"]
    else:
        result['error'] =  response.json()['error']
        
    return result