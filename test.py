from multimodal import MultiModal
sa = MultiModal("speech_ner_anonymizer")
# sa.load("test.wav")
sa.load("https://www.youtube.com/watch?v=ka6_3TJcCkA", save_folder="test_files")
sa.anonymize()
sa.export()

ss = MultiModal("speech_sentiment")
ss.load(r"test_files\Leonardo DiCaprios Powerful Climate Summit Speech.wav")
ss.get_sentiment()

sqa = MultiModal("speech-question-answering")
sqa.load(r"test_files\Leonardo DiCaprios Powerful Climate Summit Speech.wav")
sqa.get_answer("Who is Samuel?")

d2s = MultiModal("doc-to-audio")
d2s.load(r"test_files\1907.11932.pdf")
# d2s.load("Sample Text.docx")
d2s.speak()

sg = MultiModal("speech_generation")
# sg.load(r"test_files\1907.11932.pdf")
sg.load(r"test_files\Leonardo DiCaprios Powerful Climate Summit Speech.wav")
sg.listen()
sg.speak(generate=50)
