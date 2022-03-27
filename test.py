from multimodal import MultiModal
# sa = MultiModal("speech_ner_anonymizer")
# # sa.load("test_files\Leonardo DiCaprios Powerful Climate Summit Speech.wav")
# # sa.load("https://www.youtube.com/watch?v=ka6_3TJcCkA", save_folder="test_files")
# sa.load("https://www.youtube.com/watch?v=NKWKDyDKGzw", save_folder="test_files")
# sa.anonymize()
# sa.export()
#
# ss = MultiModal("speech_sentiment")
# ss.load(r"test_files\Leonardo DiCaprios Powerful Climate Summit Speech.wav")
# ss.get_sentiment()
#
# sqa = MultiModal("speech_question_answering")
# sqa.load(r"test_files\Leonardo DiCaprios Powerful Climate Summit Speech.wav")
# sqa.get_answer("Who is Samuel?")
#
# d2s = MultiModal("doc_to_audio")
# d2s.load(r"test_files\1907.11932.pdf")
# # d2s.load("Sample Text.docx")
# d2s.speak()
#
sg = MultiModal("speech_generation")
# sg.load(r"test_files\1907.11932.pdf")
sg.load(r"test_files\Leonardo DiCaprios Powerful Climate Summit Speech.wav")
sg.listen()
sg.generate(n_sentences=2)
sg.speak(generated=True)
# sg.export()
