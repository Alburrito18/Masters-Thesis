from langchain.text_splitter import RecursiveCharacterTextSplitter
import math

text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size= 100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

splitted = text_splitter.split_text(text)

[print(len(x)) for x in splitted]
print(sum([len(x) for x in splitted]), len(text))