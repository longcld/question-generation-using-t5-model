from googletrans import Translator
translator = Translator()
res = translator.translate('안녕하세요.', dest='vi').text
print(res)