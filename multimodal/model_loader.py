import os
import subprocess
import zipfile
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, AutoModelForCausalLM
from vosk import Model


class ModelLoader:
    def __init__(self, model_path):
        # if model_path and model_path != "":
        self.model_path = model_path
        self.file_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "file_info.csv"))
        self.model_dict = {'ner': self.ner_load_model, 'stt': self.vosk_load_model,
                           'sentiment-analysis': self.sent_load_model,
                           'question-answering': self.qa_load_model,
                           'text-generation': self.tg_load_model}
        # self.tasks = []

    def _get_file_info(self, task, column):
        return self.file_df[self.file_df[:]['task'] == task][column].values[0]

    def _download_load_models(self):
        self.model_folders = {task: os.path.join(self.model_path, self._get_file_info(task, 'folder')) for task in
                              self.tasks if task in self.file_df[:]['task'].values}
        self.model_urls = {task: self._get_file_info(task, 'url') for task in self.tasks if
                           task in self.file_df[:]['task'].values}
        for task in self.tasks:
            if task in self.file_df[:]['task'].values:
                self.model_dict[task]()

    def tg_load_model(self):
        if os.path.isdir(self.model_folders['text-generation']):
            self._load_transformers_model('text-generation')
        else:
            self._download_transformers_model('text-generation')
            self._load_transformers_model('text-generation')

    def qa_load_model(self):
        if os.path.isdir(self.model_folders['question-answering']):
            self._load_transformers_model('question-answering')
        else:
            self._download_transformers_model('question-answering')
            self._load_transformers_model('question-answering')

    def sent_load_model(self):
        if os.path.isdir(self.model_folders['sentiment-analysis']):
            self._load_transformers_model('sentiment-analysis')
        else:
            self._download_transformers_model('sentiment-analysis')
            self._load_transformers_model('sentiment-analysis')

    def ner_load_model(self):
        if os.path.isdir(self.model_folders['ner']):
            self._load_transformers_model('ner')
        else:
            self._download_transformers_model('ner')
            self._load_transformers_model('ner')

    def _load_transformers_model(self, task):
        tokenizer = AutoTokenizer.from_pretrained(self.model_folders[task])
        if task == 'ner':
            model = AutoModelForTokenClassification.from_pretrained(self.model_folders[task])
            self.nlp = pipeline(task, model=model, tokenizer=tokenizer)#, grouped_entities=True)
        elif task == 'sentiment-analysis':
            model = AutoModelForSequenceClassification.from_pretrained(self.model_folders[task])
            self.nlp = pipeline(task, model=model, tokenizer=tokenizer)
        elif task == 'question-answering':
            model = AutoModelForQuestionAnswering.from_pretrained(self.model_folders[task])
            self.nlp = pipeline(task, model=model, tokenizer=tokenizer)
        elif task == 'text-generation':
            model = AutoModelForCausalLM.from_pretrained(self.model_folders[task])
            self.nlp = pipeline(task, model=model, tokenizer=tokenizer)
        tokenizer, model = None, None
        return

    def _download_transformers_model(self, task):
        try:
            # Install Git - LFS on non windows
            if os.name != 'nt':
                p_git_lfs = subprocess.Popen('apt-get install git-lfs ', shell=True, stdout=subprocess.PIPE)
                p_git_lfs.wait()
                p_git_lfs.stdout.readlines()
                print("Git LFS installed.")
            git_command = 'git lfs clone ' + self.model_urls[task] + ' "' + self.model_folders[task] + '"'
            p_dl_ner = subprocess.Popen(git_command, shell=True, stdout=subprocess.PIPE)
            p_dl_ner.wait()
            p_dl_ner.stdout.readlines()
            print("Transformers NER Model downloaded.")
            return
        except Exception as e:
            print("NER Model Download failed.")

    def vosk_load_model(self):
        if os.path.isdir(self.model_folders['stt']):
            self._vosk_load_model()
        else:
            self._vosk_download_model()
            self._vosk_load_model()

    def _vosk_load_model(self):
        self.ssp = Model(self.model_folders['stt'])
        return

    def _vosk_download_model(self):
        try:
            p_dl_vosk = subprocess.Popen(
                'curl ' + self.model_urls['stt'] + ' --output "' + self.model_folders['stt'] + '.zip"',
                shell=True, stdout=subprocess.PIPE)
            p_dl_vosk.wait()
            print("VOSK Model Downloaded.")
            # p_unzip_vosk = subprocess.Popen('unzip vosk-model-en-us-0.22.zip', shell=True, stdout=subprocess.PIPE)
            # p_unzip_vosk.wait()
            folder_name = self._get_file_info('stt', 'folder')
            with zipfile.ZipFile(self.model_folders['stt'] + ".zip", 'r') as zip_ref:
                zip_ref.extractall(self.model_folders['stt'][:-len(folder_name)])
            print("VOSK Model Unzipped.")
        except Exception as e:
            print("VOSK Model Download failed.")

    def _get_ffmpeg_folder_name(self):
        ffmpeg_folder = "ffmpeg-2022-02-24-git-8ef03c2ff1-full_build"
        path = os.path.join(os.path.expanduser("~"), "multimodal", "resources")
        for i in os.listdir(path):
            if os.path.isdir(os.path.join(path, i)) and 'ffmpeg' in i:
                ffmpeg_folder = i
                break
        return ffmpeg_folder

    def _check_ffmpeg_download(self):
        ffmpeg_path = os.path.join(os.path.expanduser("~"), "multimodal", "resources",
                                   self._get_ffmpeg_folder_name(), "bin", "ffmpeg")
        if os.name == 'nt':
            ffmpeg_path += '.exe'
        if not os.path.isfile(ffmpeg_path):
            self._ffmpeg_download()
        # try:
        #     p_dl_ffmpeg = subprocess.Popen(
        #         os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources",
        #                      "ffmpeg-2022-02-24-git-8ef03c2ff1-full_build", "bin", "ffmpeg"), shell=True,
        #         stdout=subprocess.PIPE)
        #     p_dl_ffmpeg.wait()
        #     # p_out = p_dl_ffmpeg.stdout.readlines()
        #     if p_dl_ffmpeg.returncode != 0:
        #         self._ffmpeg_download()
        # except Exception as e:
        #     print("FFMPEG not found")
        #     self._ffmpeg_download()
        return

    def _ffmpeg_download(self):
        try:
            p_dl_ffmpeg = subprocess.Popen(
                'curl -L https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z --output ' + '"' +
                os.path.join(os.path.expanduser("~"), "multimodal", "resources", "ffmpeg-git-full.7z") + '"',
                shell=True, stdout=subprocess.PIPE)
            p_dl_ffmpeg.wait()
            p_dl_ffmpeg.stdout.readlines()
            print("Downloaded FFMPEG.")
        except Exception as e:
            print("Failed to download FFMPEG.")
        try:
            from pyunpack import Archive
            Archive(os.path.join(os.path.expanduser("~"), "multimodal", "resources", "ffmpeg-git-full.7z")).extractall((
                os.path.join(os.path.expanduser("~"), "multimodal", "resources")))
            print("Extracted FFMPEG Folder.")
        except Exception as e:
            print("Failed to extract FFMPEG.")
