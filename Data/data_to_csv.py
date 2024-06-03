from tika import parser
import pandas as pd
import os

def extract_text_from_pdfs_recursively(dir):
    texts = list()
    for root, _, files in os.walk(dir):
        for file in files:
            path_to_pdf = os.path.join(root, file)
            [stem, ext] = os.path.splitext(path_to_pdf)
            if ext == '.pdf':
                print("Processing " + path_to_pdf)
                pdf_contents = parser.from_file(path_to_pdf)
                texts.append(pdf_contents['content'].strip())
    return texts

if __name__ == "__main__":
    df = pd.DataFrame()
    test = extract_text_from_pdfs_recursively(os.getcwd())
    df['text'] = test
    print(df)
    print(df.iloc[1]['text'])
    