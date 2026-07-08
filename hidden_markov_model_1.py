"""
Part Of Speech (POS): Kelimelerin uygun sözcük türünü bulma çalışması.
HMM (Hidden Markov Model)

Örnek: I(Zamir/PRP) am a teacher (İsim/NN)
"""

# Kütüphaneleri içe aktar
import nltk
from nltk.tag import hmm

# Örnek eğitim verisi (training data) tanımla
# PRP: Personal Pronoun (Kişi Zamiri), VBP: Verb (Fiil), DT: Determiner (Belirteç), NN: Noun (İsim)
train_data = [
    [("I", "PRP"), ("am", "VBP"), ("a", "DT"), ("teacher", "NN")],
    [("You", "PRP"), ("are", "VBP"), ("a", "DT"), ("student", "NN")]
]

# HMM modelini eğit
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# Yeni bir cümle oluştur ve cümlenin içerisinde bulunan her bir sözcüğün türünü etiketle
test_sentence = "I am a student".split()
tags = hmm_tagger.tag(test_sentence)
print(f"Yeni Cümle Etiketleri: {tags}")

"""
Beklenen Çıktı:
Yeni Cümle Etiketleri: [('I', 'PRP'), ('am', 'VBP'), ('a', 'DT'), ('student', 'NN')]
"""

# Modelin görmediği kelimelerle test (Örnek: He is a driver)
test_sentence2 = "He is a driver".split()
tags2 = hmm_tagger.tag(test_sentence2)
print(f"Yeni Cümle Etiketleri 2: {tags2}")