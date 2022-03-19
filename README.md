
<img src="https://github.com/kritiksoman/tmp/blob/master/cover.png" width="1280" height="180"> <br>


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kritiksoman/GIMP-ML/blob/GIMP3-ML/testscases/Demo%20Notebook.ipynb)
```Python
from multimodal import MultiModal
sa = MultiModal("speech_ner_anonymizer")
# sa.load("test.wav")
sa.load("https://www.youtube.com/watch?v=ka6_3TJcCkA")
sa.anonymize()
sa.export()

ss = MultiModal("speech_sentiment")
ss.load("Leonardo DiCaprios Powerful Climate Summit Speech.wav")
ss.get_sentiment()

sqa = MultiModal("speech-question-answering")
sqa.load("Leonardo DiCaprios Powerful Climate Summit Speech.wav")
sqa.get_answer("Who is Samuel?")

d2s = MultiModal("doc-to-audio")
d2s.load("1907.11932.pdf")
# d2s.load("Sample Text.docx")
d2s.speak()

sg = MultiModal("speech_generation")
# sg.load("1907.11932.pdf")
sg.load("Leonardo DiCaprios Powerful Climate Summit Speech.wav")
sg.listen()
sg.speak(generate=50)
```


# Installation Steps
```Python
pip install multimodal
```
