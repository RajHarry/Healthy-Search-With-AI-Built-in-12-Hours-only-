import requests
import bs4
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import time
import urllib
import sys
import cv2
import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

#Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')

def img_text_model(url):
  print(">> analysis started !!!")
  t1 = time.time()
  response = requests.get(url)
  main_soup = BeautifulSoup(response.text,"html.parser")

  data = main_soup.findAll('p')
  print(">> Text Analysis")
  s = ''
  for i in data:
      s+=" "+i.getText()

  result = re.sub(r'\d+', '', s)
  result_1 = re.sub(r'[^\w\s]','',result)
  result_1 = result_1.strip()

  stop_words = set(stopwords.words('english'))
  tokens = word_tokenize(result_1.lower())
  result = [i for i in tokens if not i in stop_words]

  stemmer= PorterStemmer()
  #input_str=”There are several types of stemming algorithms.”
  #input_str=word_tokenize(result)
  res_1 = []
  for word in result:
      res_1.append(stemmer.stem(word))

  res_2 = []
  lemmatizer=WordNetLemmatizer()
  for word in res_1:
      res_2.append(lemmatizer.lemmatize(word))

  allWordExceptStopDist = nltk.FreqDist(w for w in res_2) 
  mostCommon= allWordExceptStopDist.most_common(150)
  print("time elapsed for *Text Analysis*: ",time.time()-t1)
  t1 = time.time()
  r = requests.get(url)
  html = r.text
  soup = BeautifulSoup(html, 'lxml')

  img_links = []
  for word in soup.find_all('body'):
      for k in word.findAll('img'):
        #print(k)
        #print(k['src'])
        #assert(False)
        try:
          k1 = k['src']
          img_links.append(k1)
        except:
          pass
  unique_img_list = list(dict.fromkeys(img_links))
  #plt.show()

  spam_words=['free', 'market', 'credit', 'offer', 'rate', 'remov', 'money', 'email', 'cash', 'order', 'earn', 'home', 'hidden', 'invest', 'time', 'debt', 'get', 'stock', 'claim', 'spam', 'new', 'onlin', 'dollar', 'form', 'mail', 'guarante', 'sale', 'million', 'one', 'stop', 'friend', 'busi', 'bonu', 'access', 'price', 'call', 'check', 'click', 'deal', 'today', 'per', 'incom', 'instant', 'give', 'away', 'increas', 'insur', 'lose', 'weight', 'lower', 'mortgag', 'win', 'winner', 'revers', 'age', 'asset', 'snore', 'dig', 'dirt', 'disclaim', 'statement', 'compar', 'cabl', 'convert', 'list', 'instal', 'auto', 'collect', 'lead', 'amaz', 'ad', 'promis', 'search', 'engin', 'preview', 'bureau', 'accept', 'appli', 'best', 'billion', 'brand', 'card', 'consolid', 'copi', 'dvd', 'cost', 'direct', 'dont', 'extra', 'week', 'term', 'elimin', 'e', 'financi', 'freedom', 'phone', 'prioriti', 'quot', 'sampl', 'trial', 'websit', 'refund', 'inform', 'traffic', 'request', 'internet', 'join', 'lifetim', 'limit', 'lowest', 'make', 'solut', 'hundr', 'percent', 'day', 'prize', 'refin', 'satisfact', 'isnt', 'unsecur', 'vacat', 'work', 'multi', 'level', 'wrinkl', 'compet', 'grant', 'child', 'support', 'stuff', 'tell', 'accord', 'law', 'seriou', 'satisfi', 'accordingli', 'act', 'afford', 'avoid', 'bargain', 'beneficiari', 'beverag', 'big', 'buck', 'bill', 'address', 'pager', 'buy', 'cancel', 'combin']
  h_words = ['deadli', 'bale', 'fatal', 'lethal', 'murder', 'pestil', 'imperil', 'destruct', 'damag', 'danger', 'fight', 'harm', 'deathli', 'fell', 'mortal', 'termin', 'vital', 'hostil', 'inim', 'unfriendli', 'contagi', 'infecti', 'infect', 'pestifer', 'pestilenti', 'poison', 'venom', 'insidi', 'menac', 'omin', 'sinist', 'threaten', 'hazard', 'jeopard', 'parlou', 'peril', 'riski', 'unsaf', 'unsound', 'nasti', 'noisom', 'unhealth', 'unhealthi', 'unwholesom', 'killer', 'malign', 'ruinou', 'advers', 'bad', 'bane', 'deleteri', 'detriment', 'evil', 'hurt', 'ill', 'injuri', 'mischiev', 'nocuou', 'noxiou', 'pernici', 'prejudici', 'wick', 'suicid', 'kill', 'knife', 'bomb', 'reveng', 'gun', 'weapon', 'fire', 'ak', 'effect', 'mg', 'mm', 'target', 'rifl', 'hk', 'lightweight', 'hit', 'xm', 'acsw', 'submachin', 'hunt', 'deadliest', 'cau', 'terribl', 'move', 'assault', 'barrel', 'sniper', 'grenad', 'launcher', 'defen']

  spam_count=0
  for i in mostCommon:
      if(i[0] in spam_words):
          spam_count+=1

  if(spam_count>20):
      print("WebPage Blocked (Spam Content Detected)")
  elif(spam_count>15):
    print("**Webpage Warning (Spam Content Detected)**")
  elif(spam_count>10 and spam_count<15):
      print("**Webpage Warning (Spam Content Detected)**")
      print("\n>> Image Analysis")
      weaps = ['assault_rifle','rifle','military_uniform','pickelhaube']
      il_weap = 0
      #plt.figure(1)
      for img in unique_img_list:
          urllib.request.urlretrieve(str(img), "temp_img.png")
          #load_temp_img = cv2.imread("temp_img.png")
          original = load_img("temp_img.png", target_size=(224, 224))
          numpy_image = img_to_array(original)

          image_batch = np.expand_dims(numpy_image, axis=0)
          #print('image batch size', image_batch.shape)
          #plt.imshow(np.uint8(image_batch[0]))

          processed_image = vgg16.preprocess_input(image_batch.copy())
          predictions = vgg_model.predict(processed_image)

          label = decode_predictions(predictions)
          if(label[0][0][1] in weaps):
              il_weap+=1
              #for i in range(1,il_weap+1):
              #    plt.subplot("21{}".format(i))
              #    plt.imshow(np.uint8(image_batch[0]))
      print("Illegal weapons count {} out of {} ".format(il_weap,len(unique_img_list)))
      spam_count=0
      for i in mostCommon:
          if(i[0] in h_words):
              spam_count+=1
      if(spam_count>20):
          print("** WebPage Blocked (Illegal Content) **")
      elif(spam_count>10):
          print("** Webpage Warning ((Illegal Content)) **")
      else:
          if(il_weap>3):
            print("** webpage has more illegal images **")
          else:
            print("** You are good to go!!! **")
      print("time elapsed for *Image Analysis*: ",time.time()-t1)
  else:
      #Load the VGG model        
      print("** No spam in this website.. :)**")
      print("\n>> Image Analysis")
      weaps = ['assault_rifle','rifle','military_uniform','pickelhaube']
      il_weap = 0
      #plt.figure(1)
      for img in unique_img_list:
          urllib.request.urlretrieve(str(img), "temp_img.png")
          #load_temp_img = cv2.imread("temp_img.png")
          original = load_img("temp_img.png", target_size=(224, 224))
          numpy_image = img_to_array(original)

          image_batch = np.expand_dims(numpy_image, axis=0)
          #print('image batch size', image_batch.shape)
          #plt.imshow(np.uint8(image_batch[0]))

          processed_image = vgg16.preprocess_input(image_batch.copy())
          predictions = vgg_model.predict(processed_image)

          label = decode_predictions(predictions)
          if(label[0][0][1] in weaps):
              il_weap+=1
              #for i in range(1,il_weap+1):
              #    plt.subplot("21{}".format(i))
              #    plt.imshow(np.uint8(image_batch[0]))
      print("Illegal weapons count {} out of {} ".format(il_weap,len(unique_img_list)))
      spam_count=0
      for i in mostCommon:
          if(i[0] in h_words):
              spam_count+=1
      if(spam_count>20):
          print("** WebPage Blocked (Illegal Content) **")
      elif(spam_count>10):
          print("** Webpage Warning ((Illegal Content)) **")
      else:
          if(il_weap>3):
            print("** webpage has more illegal images **")
          else:
            print("** You are good to go!!! **")
      print("time elapsed for *Image Analysis*: ",time.time()-t1)

url = input("Enter Url:")
img_text_model(url)