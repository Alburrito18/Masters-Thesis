import PyPDF2
import pandas as pd
import glob


# Specify the path to the folder
folder_path = 'socialisten'

# List all files in the folder, assuming you want to list all file types
file_path_pattern = folder_path + '/*'
files = glob.glob(file_path_pattern)

total = []
for file in files:
    with open(file, 'rb') as file:
        
        reader = PyPDF2.PdfReader(file)
        
        num_pages = len(reader.pages)
        
        #todo: add column to denote source
        for page_number in range(2,num_pages): 
            # Get the page object
            page = reader.pages[page_number]
            
            # Extract text from the page
            text = page.extract_text()
            text = text.split("?  ")
            # Print the text from the page
            text = [x.split("★") for x in text]
            
            text = filter(lambda x : len(x) == 2,text)

            # List of words to filter out
            words_to_filter = ["figur", "bild"]

            # Filter based on list of words
            text = filter(lambda x: not any(word in x[0] for word in words_to_filter), text)
            
            text = [(q,a,file.name) for q,a in text]

            total += text

        
        
        
df = pd.DataFrame(total,columns=["question","answer","file"])
df.to_parquet("qa-socialisten.parquet")
print(len(df))
            

            
