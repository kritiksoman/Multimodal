import torch
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
    if not mmtask in task_df['multimodal_task'].values:
        print("Valid tasks: " + str(list(task_df['multimodal_task'])))
        return
    base_classes = eval(str(list(task_df[task_df['multimodal_task'] == mmtask]['base']))[2:-2])
    tasks = eval(list(task_df[task_df['multimodal_task'] == mmtask]['modes'])[0])

    class MultiModalClass(*base_classes):
        def __init__(self, mmtask, tasks, model_path, vosk_logger=False):
            for base_class in base_classes:
                base_class.__init__(self, tasks, model_path)
            if vosk_logger:
                SetLogLevel(0)
            else:
                SetLogLevel(-1)
            # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.mmtask = mmtask
            self.tasks = tasks
            self._download_load_models()
            self.beep_wf = AudioSegment.from_wav(os.path.join(os.path.dirname(__file__), "resources", "beep.wav"))
            self.wf_pydub_modified, self.generated_texts, self.sentiment, self.q_answers = {}, {}, {}, {}

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

        def export(self, path=None):
            context_path, _ = self._get_input_path()
            if self.mmtask == 'speech_ner_anonymizer':
                self._export_pydub(audio_path=path)
            # elif self.mmtask == 'speech_generation':
            #     speak_text = " ".join(self.generated_texts[context_path])
            #     self._export_speak(speak_text, audio_path=path)

        def _get_input_path(self):
            if self.pdf_path:
                input_path = self.pdf_path
                input_data = self.file_doc[self.pdf_path]
            elif self.docx_path:
                input_path = self.docx_path
                input_data = self.file_doc[self.docx_path]
            # If input audio file has text
            elif len(self.doc[self.audio_path]) > 0:
                input_path = self.audio_path
                input_data = self.doc[self.audio_path]
            return input_path, input_data

        def speak(self, generated=False):
            speak_path, speak_text = self._get_input_path()
            # speak_text = self.doc[speak_path]
            # Speak existing text
            for t in speak_text:
                self._speak(t)
            if generated:
                for t in self.generated_texts[speak_path]:
                    self._speak(t)

        def generate(self, print_processing=True, prompt_context=100, n_sentences=1):
            context_path, context_text = self._get_input_path()
            # context_text = self.doc[context_path]
            # # Speak existing text
            # for t in context_text[-2:]:
            #     self._speak(t)
            if len(context_text):
                prompt_text = " ".join(context_text)
                prompt_text = prompt_text.split(" ")[-prompt_context:]
                # original_len = len(prompt_text)
                prompt_text = " ".join(prompt_text)
                self.generated_texts[context_path] = []
                while len(self.generated_texts[context_path]) < n_sentences:
                    complete_text = self.nlp(prompt_text, max_length=250)[0]['generated_text']
                    self.generated_texts[context_path] += [complete_text[len(prompt_text):]]
                    prompt_text = complete_text.split(" ")[-prompt_context:]
                    prompt_text = " ".join(prompt_text)
                if print_processing:
                    print(self.generated_texts[context_path])
            return

        def _listen(self, print_sentence=False, sentence_wise=True, task_function=None, return_result=False):
            if self.wf.getnchannels() != 1 or self.wf.getsampwidth() != 2 or self.wf.getcomptype() != "NONE":
                print("Audio file must be WAV format mono PCM. Converting format ...")
                self._convert_to_mono(file_path=self.audio_path)
                print("Conversion successful to 16 KHz mono wave file.")
                self.load()
            self.doc[self.audio_path] = []
            result = []
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
                        result.append(task_function(rec_dict))
                else:
                    pass
            rec_dict = eval(self.rec.Result())
            sentence_text = rec_dict['text']
            self.doc[self.audio_path] += [sentence_text]
            if print_sentence:
                print(sentence_text)
            if sentence_wise and task_function:
                result.append(task_function(rec_dict))
            if return_result:
                return result
            else:
                return

        def listen(self):
            self._listen()
            return

        def get_answer(self, question, print_processing=True):
            self._listen(print_sentence=print_processing, sentence_wise=False)
            answer = self.nlp(question=question, context=".".join(self.doc[self.audio_path]))
            self.q_answers[(self.audio_path,question)] = [answer]
            if print_processing:
                print(answer)
            return answer

        def get_sentiment(self, print_processing=True):
            self.sentiment[self.audio_path] = []
            self._listen(print_sentence=print_processing, task_function=lambda x: self._get_sentiment(print_processing))
            return

        def _get_sentiment(self, print_processing):
            # rec_dict = eval(self.rec.Result())
            sentiment_score = self.nlp(self.doc[self.audio_path][-1])
            self.sentiment[self.audio_path] += [sentiment_score]
            if print_processing:
                print(sentiment_score)
            return

        def anonymize(self, ner_theta=0.8, ner_window_gap=0.2, return_audio=True, print_processing=True):
            self.wf_pydub_modified[self.audio_path] = copy.deepcopy(self.wf_pydub)
            mute_rec = self._listen(print_sentence=print_processing, return_result=True,
                                    task_function=lambda x: self._mute_ner(x, ner_theta=ner_theta,
                                                                           print_processing=print_processing))
            mute_rec = [self._enlarge_window(rec, ner_window_gap) for recs in mute_rec for rec in recs]
            if len(mute_rec) > 0:
                # fix overlapping time durations
                mute_rec_fix = [(mute_rec[0][0], mute_rec[0][1])]
                mute_rec_fix += [(mute_rec[i - 1][1], mute_rec[i][1]) if mute_rec[i][0] < mute_rec[i - 1][1]
                                 else (mute_rec[i][0], mute_rec[i][1]) for i in range(1, len(mute_rec))]
                mute_start0, mute_end0 = mute_rec_fix[0]
                unmute_rec = [self.wf_pydub[:mute_start0]]
                for x in mute_rec_fix[1:]:
                    mute_start, mute_end = x
                    if mute_start - mute_end0 > 0:
                        unmute_rec += [self.wf_pydub[mute_end0:mute_start]]
                    mute_end0 = mute_end
                unmute_rec += [self.wf_pydub[mute_end0:]]
                self.wf_pydub_modified[self.audio_path] = self._mute_wf(unmute_rec)
            if return_audio:
                return self.wf_pydub_modified[self.audio_path]
            else:
                return

        def _mute_ner(self, rec_dict, ner_theta, print_processing):
            rec_ner = self.nlp(rec_dict["text"])
            rec_ner = list(filter(lambda x: x['score'] > ner_theta, rec_ner))
            rec_ner = self._merge_rec_ner(rec_dict["text"], rec_ner)
            ner_tokens = [tok for span in rec_ner for tok in span['text'].split(" ")]
            ner_tokens_rec = list(filter(lambda x: any([x['word'] in ner_tokens]), rec_dict['result']))  # change
            if print_processing:
                print(ner_tokens_rec)
            return ner_tokens_rec

    return MultiModalClass(mmtask, tasks, model_path, vosk_logger)
