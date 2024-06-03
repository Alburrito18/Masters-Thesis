import dspy
import pandas as pd

class SummarizeAsBulletedList(dspy.Module):
    def __init__(self):
        self.text_to_summary = dspy.ChainOfThought(text2summary)
        self.summary_to_bulleted_list = dspy.ChainOfThought(summary2bulleted_list)
    
    def forward(self, text: str) -> str:
        summary = self.text_to_summary(text=text).summary
        return self.summary_to_bulleted_list(summary=summary)

class text2summary(dspy.Signature):
    #"""Summarizes a given text."""
    """Summerar en given text."""

    text = dspy.InputField()
    summary = dspy.OutputField()

class summary2bulleted_list(dspy.Signature):
    #"""Takes a summary and converts it to a bulleted list. 
    #The list has three topics: 'Medical History', 'Reasons for getting Care' & 'Measures taken by the caregiver'"""
    """Tar en summering och konverterar den till en punktlista.
    Listan har tre ämnen: '1. Sjukdomshistoria', '2. Sökorsaker' och '3. Åtgärder'."""
    
    summary = dspy.InputField()
    bulleted_list = dspy.OutputField()


#if there are errors finding the file, run terminal in top level dir
df = pd.read_parquet("OpenAI/synthetic_229_corrected.parquet")
text = df.iloc[0]["description"]
print("### Gold Standard Summary ###\n",df.iloc[0]["summary"],"\n\n")

gpt4 = dspy.OpenAI(model="gpt-4", max_tokens=2000, model_type="chat")
dspy.settings.configure(lm=gpt4)

summary = SummarizeAsBulletedList()(text).bulleted_list
print("### Candidate Summary ###\n",summary)