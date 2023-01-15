import torchaudio
import torch

import numpy as np
import telebot

from model import M5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bot = telebot.TeleBot("")

@bot.message_handler(commands = ['start'])
def send_welcome(message):
    bot.reply_to(message, 'Hello and Welcome!\n Send me a voice please')

@bot.message_handler(content_types = ['voice'])
def voice_processing(message):
  myvoice = bot.get_file(message.voice.file_id)
  myfile = bot.download_file(myvoice.file_path)
  voicepath = myvoice.file_path

  with open(voicepath, 'wb') as audio:
    audio.write(myfile)

  signal, sample_rate = torchaudio.load(voicepath)



  model = M5(n_output = 11).to(device)
  model = model.to(device)
  model.load_state_dict(torch.load('weights.pth'))

  model.eval()

  signal = torch.mean(signal, dim = 0, keepdim = True)
  transform = torchaudio.transforms.Resample(sample_rate, 8000)
  signal = transform(signal)

  tensor = signal.unsqueeze(0).to(device)

  preds = model(tensor)

  preds = preds.cpu().detach().numpy()

  labels = ['alireza', 'amir', 'benyamin', 'hossein', 'maryam', 'mohammad', 'morteza', 'nahid', 'parisa', 'zahra', 'zeynab']
  output = np.argmax(preds)

  bot.reply_to(message, f'You are {labels[output]} 🧐')

bot.polling()