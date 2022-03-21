from pydub import AudioSegment
from vosk import KaldiRecognizer, SetLogLevel
from .audio import Audio
from .text import Text
import copy
import os
import pandas as pd


def MultiModal(mmtask, model_path=os.path.join(os.path.expanduser("~"), "multimodal", "resources"), vosk_logger=False):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    task_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "tasks.tsv"), sep='\t')
    base_classes = eval(str(list(task_df[task_df['multimodal_task'] == mmtask]['base']))[2:-2])
    tasks = eval(list(task_df[task_df['multimodal_task'] == mmtask]['modes'])[0])

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

        def load(self, path=None, max_pages=2, page_numbers=None, save_folder=None):
            if path and "speech" in mmtask:
                self._load_audio(path, save_folder)
                print("Read speech file.")
            elif path and "doc" in mmtask:
                if path.endswith('pdf'):
                    self._load_pdf(path, maxpages=max_pages, page_numbers=page_numbers)
                else:
                    self._load_docx(path)
                print("Read doc file.")

        def speak(self, text=None, generate=0, prompt_context=100):
            if isinstance(text, str):
                text = [text]
            if text:
                speak_text = text
            # If input file has text
            elif self.pdf_path:
                speak_text = self.file_doc[self.pdf_path]
            elif self.docx_path:
                speak_text = self.file_doc[self.docx_path]
            # If input audio file has text
            elif len(self.doc[self.audio_path]) > 0:
                speak_text = self.doc[self.audio_path]
            # Speak existing text
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

        def _listen(self, print_sentence=False, sentence_wise=True, task_function=None):
            if self.wf.getnchannels() != 1 or self.wf.getsampwidth() != 2 or self.wf.getcomptype() != "NONE":
                print("Audio file must be WAV format mono PCM. Converting format ...")
                self._convert_to_mono(file_path=self.audio_path)
                print("Conversion successful to 16 KHz mono wave file.")
                self.load()
            self.doc[self.audio_path] = []
            while True:
                data = self.wf.readframes(4000)
                if len(data) == 0:
                    break
                if self.rec.AcceptWaveform(data):
                    rec_dict = eval(self.rec.Result())
                    sentence_text = rec_dict['text']
                    if print_sentence:
                        print(sentence_text)
                    self.doc[self.audio_path] += [sentence_text]
                    if sentence_wise and task_function:
                        task_function(rec_dict)
                else:
                    pass
            rec_dict = eval(self.rec.Result())
            sentence_text = rec_dict['text']
            self.doc[self.audio_path] += [sentence_text]
            if print_sentence:
                print(sentence_text)
            if sentence_wise and task_function:
                task_function(rec_dict)
            return

        def listen(self):
            self._listen()
            return

        def get_answer(self, question):
            self._listen(sentence_wise=False)
            answer = self.nlp(question=question, context=".".join(self.doc[self.audio_path]))
            print(answer)
            return answer

        def get_sentiment(self):
            self._listen(print_sentence=True, task_function=self._get_sentiment)
            return

        def _get_sentiment(self, rec_dict):
            # rec_dict = eval(self.rec.Result())
            sentiment_score = self.nlp(self.doc[self.audio_path][-1])
            print(sentiment_score)
            return

        def anonymize(self, ner_theta=0.8, ner_window_gap=0.1, return_audio=True):
            self._listen(print_sentence=True, task_function=lambda x: self._mute_ner(x, ner_theta=ner_theta,
                                                                                     ner_window_gap=ner_window_gap))
            if return_audio:
                return self.wf_pydub_modified
            else:
                return


        def _mute_ner(self, rec_dict, ner_theta, ner_window_gap):
            # rec_dict = eval(self.rec.Result())
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
