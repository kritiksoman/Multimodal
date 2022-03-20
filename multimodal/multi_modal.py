from pydub import AudioSegment
from vosk import KaldiRecognizer, SetLogLevel
from .audio import Audio
from .text import Text
import copy
import os


def MultiModal(mmtask, model_path=os.path.join(os.path.expanduser("~"), "multimodal", "resources"), vosk_logger=False):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if mmtask == "speech_ner_anonymizer":
        base_classes = Text, Audio
        tasks = ["ner", "stt"]

    elif mmtask == "speech_sentiment":
        base_classes = Text, Audio
        tasks = ["sentiment-analysis", "stt"]

    elif mmtask == "speech-question-answering":
        base_classes = Text, Audio
        tasks = ["question-answering", "stt"]

    elif mmtask == "doc-to-audio":
        base_classes = Text, Audio
        tasks = ["speak"]

    elif mmtask == "speech_generation":
        base_classes = Text, Audio
        tasks = ["stt", "text-generation", "speak"]

    class MultiModalClass(*base_classes):
        def __init__(self, tasks, model_path, vosk_logger=False):
            for base_class in base_classes:
                base_class.__init__(self, tasks, model_path)
            if vosk_logger:
                SetLogLevel(0)
            else:
                SetLogLevel(-1)
            self.tasks = tasks
            self._download_load_models()
            self.beep_wf = AudioSegment.from_wav(os.path.join(os.path.dirname(__file__), "resources", "beep.wav"))
            self.wf_pydub_modified = None

        def load(self, path=None, maxpages=2, page_numbers=None, save_folder=None):
            if path and "speech" in mmtask:
                self._load_audio(path, save_folder)
            elif path and "doc" in mmtask:
                if path.endswith('pdf'):
                    self._load_pdf(path, maxpages=maxpages, page_numbers=page_numbers)
                else:
                    self._load_docx(path)
                print("read doc file.")

        def speak(self, text=None, generate=0, prompt_context=20):
            if isinstance(text, str):
                text = [text]
            if text:
                speak_text = text
            elif self.pdf_path:
                speak_text = self.file_doc[self.pdf_path]
            elif self.docx_path:
                speak_text = self.file_doc[self.docx_path]
            elif len(self.doc[self.audio_path]) > 0:
                speak_text = self.doc[self.audio_path]
            for t in speak_text[-2:]:
                self._speak(t)
            if generate:
                prompt_text = " ".join(speak_text)
                prompt_text = prompt_text.split(" ")[-prompt_context:]
                # original_len = len(prompt_text)
                prompt_text = " ".join(prompt_text)
                generated_texts = []
                while len(" ".join(generated_texts).split(" ")) < generate:
                    complete_text = self.nlp(prompt_text, max_length=250)[0]['generated_text']
                    generated_texts += [complete_text[:len(prompt_text)]]
                    prompt_text = complete_text.split(" ")[-prompt_context:]
                    prompt_text = " ".join(prompt_text)
                print(generated_texts)
                for t in generated_texts:
                    self._speak(t)

        def listen(self):
            if self.wf.getnchannels() != 1 or self.wf.getsampwidth() != 2 or self.wf.getcomptype() != "NONE":
                print("Audio file must be WAV format mono PCM. Converting format ...")
                self._convert_to_mono(file_path=self.audio_path)
                print("Conversion successful to 16 KHz mono wave file.")
                self.load()
            self.doc[self.audio_path] = []
            # self.wf_pydub_modified = copy.deepcopy(self.wf_pydub)
            while True:
                data = self.wf.readframes(4000)
                if len(data) == 0:
                    break
                if self.rec.AcceptWaveform(data):
                    sentence_text = eval(self.rec.Result())['text']
                    # if len(self.doc):
                    self.doc[self.audio_path] += [sentence_text]
                    # else:
                    #     self.doc[self.audio_path] = [sentence_text]
                else:
                    pass
            sentence_text = eval(self.rec.Result())['text']
            # if len(self.doc):
            self.doc[self.audio_path] += [sentence_text]
            # else:
            #     self.doc[self.audio_path] = [sentence_text]
            print(self.doc[self.audio_path])

        def get_answer(self, question):
            self.listen()
            # if self.wf.getnchannels() != 1 or self.wf.getsampwidth() != 2 or self.wf.getcomptype() != "NONE":
            #     print("Audio file must be WAV format mono PCM. Converting format ...")
            #     self._convert_to_mono(file_path=self.audio_path)
            #     print("Conversion successful to 16 KHz mono wave file.")
            #     self.load()
            # self.doc[self.audio_path] = []
            # self.wf_pydub_modified = copy.deepcopy(self.wf_pydub)
            # while True:
            #     data = self.wf.readframes(4000)
            #     if len(data) == 0:
            #         break
            #     if self.rec.AcceptWaveform(data):
            #         sentence_text = eval(self.rec.Result())['text']
            #         if len(self.doc):
            #             self.doc[self.audio_path] += [sentence_text]
            #     else:
            #         pass
            # sentence_text = eval(self.rec.Result())['text']
            # if len(self.doc):
            #     self.doc[self.audio_path] += [sentence_text]
            answer = self.nlp(question=question, context=".".join(self.doc[self.audio_path]))
            print(answer)
            return answer

        def get_sentiment(self):
            if self.wf.getnchannels() != 1 or self.wf.getsampwidth() != 2 or self.wf.getcomptype() != "NONE":
                print("Audio file must be WAV format mono PCM. Converting format ...")
                self._convert_to_mono(file_path=self.audio_path)
                print("Conversion successful to 16 KHz mono wave file.")
                self.load()
            self.doc[self.audio_path] = []
            # self.wf_pydub_modified = copy.deepcopy(self.wf_pydub)
            while True:
                data = self.wf.readframes(4000)
                if len(data) == 0:
                    break
                if self.rec.AcceptWaveform(data):
                    sentiment_score = self._get_sentiment()
                    # if len(self.doc):
                    self.doc[self.audio_path] += [sentiment_score]
                else:
                    pass

            sentiment_score = self._get_sentiment()
            self.doc[self.audio_path] += [sentiment_score]
            return

        def _get_sentiment(self):
            rec_dict = eval(self.rec.Result())
            sentiment_score = self.nlp(rec_dict["text"])
            print(rec_dict["text"], sentiment_score)
            return sentiment_score

        def anonymize(self, ner_theta=0.8, ner_window_gap=0.1, return_audio=True):
            if self.wf.getnchannels() != 1 or self.wf.getsampwidth() != 2 or self.wf.getcomptype() != "NONE":
                print("Audio file must be WAV format mono PCM. Converting format ...")
                self._convert_to_mono(file_path=self.audio_path)
                print("Conversion successful to 16 KHz mono wave file.")
                self.load()
            self.doc[self.audio_path] = []
            self.wf_pydub_modified = copy.deepcopy(self.wf_pydub)
            while True:
                data = self.wf.readframes(4000)
                if len(data) == 0:
                    break
                if self.rec.AcceptWaveform(data):
                    result_dict = self._mute_ner(ner_theta, ner_window_gap)
                    print(result_dict['text'])
                    if len(self.doc):
                        self.doc[self.audio_path] += [result_dict['text']]
                else:
                    pass

            result_dict = self._mute_ner(ner_theta, ner_window_gap)
            print(result_dict['text'])
            if len(self.doc):
                self.doc[self.audio_path] += [result_dict['text']]
            if return_audio:
                return self.wf_pydub_modified
            else:
                return

        def _mute_ner(self, ner_theta, ner_window_gap):
            rec_dict = eval(self.rec.Result())
            rec_ner = self.nlp(rec_dict["text"])
            rec_ner = list(filter(lambda x: x['score'] > ner_theta, rec_ner))
            rec_ner = self._merge_rec_ner(rec_dict["text"], rec_ner)
            ner_tokens = [tok for span in rec_ner for tok in span['text'].split(" ")]
            ner_tokens_rec = list(filter(lambda x: any([x['word'] in ner_tokens]), rec_dict['result']))  # change
            print(ner_tokens_rec)
            if len(ner_tokens_rec) > 0:
                aud_dict = ner_tokens_rec[0]
                mute_start0, mute_end0 = self._enlarge_window(aud_dict, ner_window_gap)
                mute_start, mute_end = copy.deepcopy(mute_start0), copy.deepcopy(mute_end0)
                unmute_rec = [self.wf_pydub[:mute_start0]]
                for x in ner_tokens_rec[1:]:
                    aud_dict = x
                    mute_start, mute_end = self._enlarge_window(aud_dict, ner_window_gap)
                    unmute_rec += [self.wf_pydub[mute_end0:mute_start]]
                    mute_end0 = mute_start
                unmute_rec += [self.wf_pydub[mute_end:]]
                self.wf_pydub_modified = self._mute_wf(unmute_rec)
            return rec_dict

    return MultiModalClass(tasks, model_path, vosk_logger)
