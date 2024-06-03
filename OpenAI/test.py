from openai import OpenAI

import os
import sys

# Get the directory of the current file (test.py) and then add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now you can import util or any of its functions directly
from util import open_mock_data

raw_df, _, summary_df = open_mock_data()

X = str(raw_df.iloc[0]["metadata"]) + raw_df.iloc[0]["text"]
y = summary_df.iloc[0]["text"]

#print(X,"\n\n",y)
#todo: remove xlsx new lines

client = OpenAI()

instructions = """You are a helpful assistant, that helps that create synthetic data. 
                  You will be provided with a long description of a patient and a summary.
                  The description and the summary are both in swedish, and you should
                  answer in swedish.
                """
question = f"""Please create a similar patient <Description>{X}</Description><Summary>{y}</Summary>"""
messages = [{"role": "system", "content": instructions}, {"role": "user", "content": question}]

response = client.chat.completions.create(
  model="gpt-4-0125-preview",
  #response_format={ "type": "json_object" },
  messages=messages
)
print(response.choices[0].message.content)

# Get answer

#response = openai.chat.completions.create() #ChatCompletion.create(model=bot, messages=messages)
#print(response)
#response = response["choices"][0]["message"]
#answer = response["content"]
#print(answer)