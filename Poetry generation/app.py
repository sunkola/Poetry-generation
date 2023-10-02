import requests
import re
import random
import json
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import Entry, Label, Button, Text, Scrollbar, StringVar
from tensorflow.keras.models import Sequential, load_model


model = load_model("./my_model.h5")

def cofe_poem(seed_text, col):
    f = open("./wordict.json","r")
    word_index_dict = json.load(f)
    f.close()
    
    f = open("./inddict.json","r")
    index_dict = json.load(f)
    f.close()
#     print(index_dict)
    rows = 4
    cols = col+1
    seq_len = 2
    chars = re.findall('[\x80-\xff]{3}|[\w\W]', seed_text)
    if len(chars) != seq_len:
        return ""
    arr = [word_index_dict[k] for k in chars]
    for i in range(seq_len, rows * cols):
        if (i+1) % cols == 0:
            if (i+1) / cols == 2 or (i+1) / cols == 4:
                arr.append(2)
            else:
                arr.append(1)
        elif (i+1) ==3 or (i+1) % cols == 1:
            proba = model.predict(np.reshape(np.array(arr[-seq_len:]), (1, seq_len)), verbose=0)
            predicted = np.argsort(proba[0])[-64:]
            index = random.randint(0,len(predicted)-1)
            new_char = predicted[index]
            while new_char == 1 or new_char == 2 or new_char == arr[-2] or new_char == arr[-1]:
                index = random.randint(0,len(predicted)-1)
                new_char = predicted[index]
            arr.append(new_char) 
        else:
            proba = model.predict(np.reshape(np.array(arr[-seq_len:]), (1, seq_len)), verbose=0)
            predicted = np.argsort(proba[0])[-8:]
            index = random.randint(0,len(predicted)-1)
            new_char = predicted[index]
            while new_char == 1 or new_char == 2 or new_char == arr[-2] or new_char == arr[-1]:
                index = random.randint(0,len(predicted)-1)
                new_char = predicted[index]
            arr.append(new_char)
    poem = [index_dict[str(i)] for i in arr]
    return "".join(poem)

class PoetryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("詩詞生成器")
        self.geometry("500x500")
        
        self.init_ui()
        
    def init_ui(self):
        self.label1 = Label(self, text="請輸入起始文字:")
        self.label1.pack(pady=10)
        
        self.entry_seed = Entry(self)
        self.entry_seed.pack(pady=10)

        self.label2 = Label(self, text="請輸入字詞數量:")
        self.label2.pack(pady=10)
        
        self.entry_length_var = StringVar()
        self.entry_length = Entry(self, textvariable=self.entry_length_var)
        self.entry_length.pack(pady=10)
        self.entry_length_var.set("4") 
        
        self.gen_button = Button(self, text="生成詩詞", command=self.generate_poem)
        self.gen_button.pack(pady=10)
        
        self.text_area = Text(self, height=15, width=40)
        self.text_area.pack(pady=10)
        
        self.scrollbar = Scrollbar(self, command=self.text_area.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text_area.configure(yscrollcommand=self.scrollbar.set)
        
    def generate_poem(self):
        seed_text = self.entry_seed.get()
        length = int(self.entry_length_var.get())
        generated_poem = cofe_poem(seed_text, length)
        self.text_area.delete(1.0, tk.END) 
        self.text_area.insert(tk.END, generated_poem)

if __name__ == "__main__":
    app = PoetryApp()
    app.mainloop()
