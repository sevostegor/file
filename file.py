import tkinter as tk

import tkinter
import numpy as np
import tensorflow
import collections
import re
import os
import pandas as pd

from tensorflow.keras.layers import Dense, Embedding, LSTM, SimpleRNN, Input, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

global marker
marker = 0


def define(ask):
        skills[ask]()

   

def rnn_text():

  global marker

  file = pd.read_csv('rnn.csv')
  news = file['headline_text'].tolist()
  slicer = len(news)// 500
  news = news[:slicer]
  text = ''
  for i in news:
    text += i + ', '
  maxWords = 2000
  texts = text
  tokenizer = Tokenizer(num_words=maxWords, filters='!–"—#$%&;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                        lower=True, split=' ', char_level=False)
  tokenizer.fit_on_texts([texts])
  dist = list(tokenizer.word_counts.items())
  print(dist)
  inp_words = 3
  data = tokenizer.texts_to_sequences([texts])
  res = np.array(data[0])
  n = res.shape[0]-inp_words
  X = np.array([res[i:i+inp_words] for i in range(n)])
  Y = to_categorical(res[inp_words:], num_classes=maxWords)



  if os.path.exists('saved_model_rnn.keras') == False:
      model = Sequential()
      model.add(Embedding(maxWords, 256, input_length=inp_words))
      model.add(SimpleRNN(120, activation='tanh'))
      model.add(Dropout(0.4))
      model.add(Dense(maxWords, activation='softmax'))
      model.summary()

      model.compile(loss='categorical_crossentropy', optimizer='Nadam')
      history = model.fit(X, Y, batch_size=100, epochs=10)

  else:
      model = load_model('saved_model_rnn.keras')
  


  def buildPhrase(texts, str_len = 4):
    res = texts
    data = tokenizer.texts_to_sequences([texts])[0]
    for i in range(str_len):
      x = data[i: i + inp_words]
      inp = np.expand_dims(x, axis=0)

      pred = model.predict( inp )
      indx = pred.argmax(axis=1)[0]
      data.append(indx)

      res += " " + tokenizer.index_word[indx]

    return res

  say = str(x_text.get("1.0", "end"))
  print(66666666666)
  add(buildPhrase(say))



def lstm():
  file = pd.read_csv('lstm.csv')
  passengers = file['#Passengers'].tolist()
  data = passengers
  ress = np.array(data)
  inp_num = 3
  n = len(ress) -inp_num
  x_train = np.array([ress[i:i+inp_num] for i in range(n)])
  y_train = [ress[inp_num:]]



  if os.path.exists('saved_model_lstm.keras') == False:
      model = Sequential()
      model.add(LSTM(128, return_sequences=True, input_shape=[x_train.shape[1], 1]))
      model.add(LSTM(128, return_sequences=False))
      model.add(Dense(1))
      model.compile(optimizer='adam', loss = 'MAPE')
      history = model.fit(x_train, y_train, batch_size = 3, epochs=60)
      model.save('saved_model_lstm.keras')


      def buildSequence(number, str_len = 3):
        res = []
        data = number
        for i in range(str_len):
          x = data[i: i + inp_num]
          inp = np.array(x).reshape(1,3,1)
  

          pred = model.predict( inp )
        
          tolist = pred.tolist()[0][0]
        
          data.append(int(tolist))
          

          res.append(int(tolist))
          tolist = int(tolist)

        return res

      arr = list((map(int, x_text.get(1.0, 3.0).split())))
      add(buildSequence(arr))




  elif os.path.exists('saved_model_lstm.keras') == True:

      model = load_model('saved_model_lstm.keras')
      def buildSequence(number, str_len = 3):
        res = []
        data = number
        for i in range(str_len):
          x = data[i: i + inp_num]
          inp = np.array(x).reshape(1,3,1)
          pred = model.predict( inp )
          tolist = pred.tolist()[0][0]
          data.append(int(tolist))
          res.append(int(tolist))
          tolist = int(tolist)

        return res
    
      arr = list((map(int, x_text.get(1.0, 3.0).split())))
      add(buildSequence(arr))

def simple_nn():
    file = pd.read_csv('simple_nn.csv')
    file.dropna()
    x = file['Pclass'].tolist()
    y = file['Survived'].tolist()
  

    if os.path.exists('saved_model_snn.keras') == False:
        model = Sequential()
        model.add(Dense(units = 50, input_shape = [1]))
        model.add(Dense(units = 25))
        model.add(Dense(units = 1))
        model.compile(loss='MAE', optimizer='adam')
        history = model.fit(x = x, y = y, epochs = 30)
        model.save('saved_model_snn.keras')

        input_arr = list((map(float, x_text.get(0.0).split())))
        predicted = model.predict(input_arr)
        add(predicted)
    else:
        model = load_model('saved_model_snn.keras')
        input_arr = list((map(float, input().split())))
        predicted = model.predict(input_arr)
        add(predicted)
        marker = True

def main():
  global exit_marker
  inputt = arc
  define(inputt)

skills = {'Написание текста': rnn_text,

          'Долгосрочный прогноз': lstm,

          'Классификация': simple_nn}












def arc_def():
    global arc
    arc = rb_var.get()
    if arc == 1:
        arc = 'Классификация'
    elif arc == 2:
        arc = 'Долгосрочный прогноз'
    elif arc == 3:
        arc = 'Написание текста'
def execute():
    main()
def add(addd):
    y_text.insert(0.0, addd)
    
win = tk.Tk()
win.geometry('650x650')
win.title('Project')

x_label = tk.Label(win, text = 'Входные значения:', font = ('Arial', 15))
x_label.grid(row = 0, column = 0, padx = 5, pady = 5)


x_text = tk.Text(win, font = ('Arial', 15), height = 5, width = 20)
x_text.grid(row = 1, column = 0, padx = 30, pady = 10)
win.grid_columnconfigure(0, minsize = 4)
win.grid_rowconfigure(1, minsize = 100)


y_label = tk.Label(win, text = 'Выходные значения:', font = ('Arial', 15))
y_label.grid(row = 0, column = 1, padx = 5, pady = 5)


y_text = tk.Text(win, font = ('Arial', 15), height = 5, width = 20)
y_text.grid(row = 1, column = 1, padx = 30, pady = 10)
win.grid_columnconfigure(1, minsize = 100)

rb_label = tk.Label(win, text = 'Ваша задача:', font = ('Arial', 15), height = 5, width = 20)
rb_label.grid(row = 2, column = 0, padx = 30, pady = 10)

rb_var = tk.IntVar()

rb_1 = tk.Radiobutton(win, text = 'Классификация', font = ('Arial', 12), variable = rb_var, value = 1, command =arc_def)
rb_1.grid(row = 3, column = 0, padx = 30, pady = 10, sticky = 'W')
rb_2 = tk.Radiobutton(win, text = 'Долгосрочный прогноз значений', font = ('Arial', 12), variable = rb_var, value = 2, command =arc_def)
rb_2.grid(row = 4, column = 0, padx = 30, pady = 10, sticky = 'W')
rb_3 = tk.Radiobutton(win, text = 'Генерация текста', font = ('Arial', 12), variable = rb_var, value = 3, command =arc_def)
rb_3.grid(row = 5, column = 0, padx = 30, pady = 10, sticky = 'W')

btn_exec = tk.Button(win, text = 'Получить данные', font = ('Arial', 15), height = 1, width = 20, command = execute)
btn_exec.grid(row = 5, column = 1, padx = 30, pady = 10)


win.mainloop()
print(arc)
