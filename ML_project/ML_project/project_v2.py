from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import customtkinter
import tkinter
import numpy as np
from textblob import TextBlob
from matplotlib import pyplot as plt
import seaborn as sns


app = customtkinter.CTk()
app.geometry("600x500")
app.title("Twitter Sentiment Analyser")
app.iconbitmap('C:/Users/lina/Desktop/ML_project/birdt.ico')

frame_2 = customtkinter.CTkFrame(master=app)
frame_2.pack(fill="both", expand=True)
frame_2.place(x=0, y=0)

frame_1 = customtkinter.CTkFrame(master=app)
frame_1.pack(pady=20, padx=60, fill="both", expand=True)



bg = tkinter.PhotoImage(file='C:/Users/lina/Desktop/ML_project/11.png')
bg_label = customtkinter.CTkLabel(frame_2, image=bg, text=None)
bg_label.place(x=0, y=0)
bg_label.pack()


entry_1 = customtkinter.CTkTextbox(master= frame_1, height= 100, width=380)
entry_1.place(y= 100, x=50)

def temp_text(e):
    entry_1.delete("1.0","end")
entry_1.insert("1.0", "Write the tweet here:  ")
entry_1.bind("<FocusIn>", temp_text)





def pre():
    tweet = entry_1
    
    # precprcess tweet
    tweet_words = []
    tweet2 = entry_1.get("1.0",'end-1c')

    for word in tweet2.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)
    # load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    # label_2= customtkinter.CTkLabel(master=frame_1, text="The Tweet is: ")
    labels = ['Negative', 'Neutral', 'Positive']

    # sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    output_1.delete('1.0', tkinter.END)

    for i in range(len(scores)):
        
        l = labels[i]
        s = scores[i]
        print(l,s)
        output_1.insert("end", f"{l} {s}\n")
        

    output_2.delete('1.0', tkinter.END)        
    m=max(scores)
    ind=np.where(scores==m)[0]
    lab=labels[ind[0]]
    output_2.insert("end", f"{lab} ")
    return lab


def analyze():
    tweet = entry_1
    
    # precprcess tweet
    tweet_words = []
    tweet2 = entry_1.get("1.0",'end-1c')

    for word in tweet2.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)
    # load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    # label_2= customtkinter.CTkLabel(master=frame_1, text="The Tweet is: ")
    labels = ['Negative', 'Neutral', 'Positive']

    # sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    output_1.delete('1.0', tkinter.END)

    for i in range(len(scores)):
        
        l = labels[i]
        s = scores[i]
        print(l,s)
        output_1.insert("end", f"{l} {s}\n")
        

    output_2.delete('1.0', tkinter.END)        
    m=max(scores)
    ind=np.where(scores==m)[0]
    lab=labels[ind[0]]
    output_2.insert("end", f"{lab} ")
    
    labels = ['Negative', 'Neutral', 'Positive']
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    plt.figure(figsize=(7,7))
    plt.pie(scores, labels= labels, startangle=0, explode=[0.1, 0.1, 0.1], colors=["#e95959", "#fcb42c", "#8ec21f"], autopct="%2.1f%%")
    plt.legend(title= "Tweet Analysis Result")
    plt.show()

# "red", "orange", "green"

def clear():
    entry_1.delete('1.0', tkinter.END)
    output_1.delete('1.0', tkinter.END)
    output_2.delete('1.0', tkinter.END)


buutton_1 = customtkinter.CTkButton(master=frame_1, command=pre, text="Predict", width=110)
buutton_1.pack(pady=12, padx=10)
buutton_1.place(y=250, x=50)


buutton_2 = customtkinter.CTkButton(master=frame_1, text="Clear", command= clear, width=110)
buutton_2.pack(pady=12, padx=10)
buutton_2.place(y=250, x=310)


buutton_3 = customtkinter.CTkButton(master=frame_1, text="Visualization", command= analyze, width=110)
buutton_3.pack(pady=12, padx=10)
buutton_3.place(y=250, x=180)


output_1 = customtkinter.CTkTextbox(master= frame_1, height= 100, width=380)
output_1.place(y= 330, x=50)


output_2 = customtkinter.CTkTextbox(master= frame_1, height= 20, width=100,text_color='white', fg_color="transparent")
output_2.place(y= 300, x=127)


label_1 = customtkinter.CTkLabel(master=frame_1, justify=tkinter.CENTER, text="Welcome To Twitter Sentiment Analyses System!", font=("Tekton Pro Cond", 18), text_color="#fff9d9")
label_1.pack(pady=12, padx=10)
label_1.place(y=20, x=53)


smallimage = tkinter.PhotoImage(file="C:/Users/lina/Desktop/ML_project/logo-twitter-trans.png")
smallimagelabel = customtkinter.CTkLabel(master=frame_1, image= smallimage, text=None)
smallimagelabel.place(y=5, x=5)


label_2 = customtkinter.CTkLabel(master=frame_1, justify=tkinter.LEFT, text="The tweet is: "  , font=("Seaford Display",15), text_color="#fff9d9")
label_2.pack(pady=12, padx=10)
label_2.place(y=300, x=50)


app.mainloop()