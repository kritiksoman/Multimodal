import os
import re
import subprocess
import wave
from pydub import AudioSegment
import numpy as np
import youtube_dl
from vosk import KaldiRecognizer
from .model_loader import ModelLoader
import pyttsx3


class Audio(ModelLoader):
    def __init__(self, tasks, model_path):
        self.wf = None
        self.wf_pydub = None
        self.wf_pydub_modified = None
        self.rec = None
        self.audio_path = None
        self.url = None
        super(Audio, self).__init__(model_path)
        # self.tasks = ["sst"]

    def _load_audio(self, audio_path=None, save_folder=None):
        # Download Audio from YouTube if URL is provided
        if re.match("(http(s)??\:\/\/)?(www\.)?((youtube\.com\/watch\?v=)|(youtu.be\/))([a-zA-Z0-9\-_])+", audio_path):
            filename = self._get_audio_youtube(audio_path, save_folder)
            self.url = audio_path
            self.audio_path = filename
        elif os.path.exists(audio_path):
            self.audio_path = audio_path
        # elif os.path.exists(os.path.join(os.path.abspath("."), audio_path)):
        #     self.audio_path = os.path.join(os.path.abspath("."), audio_path)
        # Change format, update audio_path, convert to mono
        if self.audio_path.endswith(".mp3"):
            self._check_ffmpeg_download()
            self._convert_ffmpeg()
            self.audio_path = self.audio_path[:-4] + ".wav"
            self._convert_to_mono(self.audio_path)
        # Load 16KHz audio file
        self.wf = wave.open(self.audio_path, "rb")
        self.wf_pydub = AudioSegment.from_wav(self.audio_path)
        if os.name == 'nt':
            os.environ['path'] += ';' + os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources",
                                                     self._get_ffmpeg_folder_name())
        self.rec = KaldiRecognizer(self.ssp, self.wf.getframerate())
        self.rec.SetWords(True)

    def _convert_ffmpeg(self):
        # example
        # ffmpeg-2022-02-24-git-8ef03c2ff1-full_build\\bin\\ffmpeg -i test.mp3 test.wav -y
        # os.path.dirname(__file__)
        if os.name == 'nt':
            ffmpeg_folder = self._get_ffmpeg_folder_name()
            convert_command = '"'+os.path.join(os.path.expanduser("~"), "multimodal", 'resources', ffmpeg_folder, 'bin', 'ffmpeg') + '" -i "' + self.audio_path + '" "' + self.audio_path[:-4] + '.wav" -y'
            p_convert_ffmpeg = subprocess.Popen(convert_command, shell=True, stdout=subprocess.PIPE)
            p_convert_ffmpeg.wait()
            p_convert_ffmpeg.stdout.readlines()
            print("File converted to wav using FFMPEG.")
        else:
            convert_command = 'ffmpeg -i "' + self.audio_path + '" "' + self.audio_path[:-4] + '.wav" -y'
            p_convert_ffmpeg = subprocess.Popen(convert_command, shell=True, stdout=subprocess.PIPE)
            p_convert_ffmpeg.wait()
            p_convert_ffmpeg.stdout.readlines()
            print("File converted to wav using FFMPEG.")

    def _convert_to_mono(self, file_path=None, frequency=16000):
        # try:
        if file_path is None:
            sound = self.wf_pydub
        else:
            sound = AudioSegment.from_wav(file_path)
        sound.set_channels(1)
        sound = sound.set_frame_rate(frequency)
        sound = sound.set_channels(1)
        if file_path is None:
            self.wf_pydub = sound
        else:
            sound.export(file_path, format="wav")
        # except Exception as e:
        #     print(e)

    def _enlarge_window(self, aud_dict, window_gap=0.1):
        return (int(np.floor((aud_dict['start'] - window_gap) * 1000)), int(
            np.ceil((aud_dict['end'] + window_gap) * 1000)))

    def _get_audio_youtube(self, video_url, save_folder=None):
        video_info = youtube_dl.YoutubeDL().extract_info(
            url=video_url, download=False
        )
        # filename can have only space, period or alphanumeric characters
        filename = ''.join(c for c in f"{video_info['title']}.mp3" if c.isalnum() or c in [" ", "."])
        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_folder = os.path.abspath(save_folder)
            filename = os.path.join(save_folder, filename)
        options = {
            'format': 'bestaudio/best',
            'keepvideo': False,
            'outtmpl': filename,
        }
        with youtube_dl.YoutubeDL(options) as ydl:
            ydl.download([video_info['webpage_url']])
        print("Download complete... {}".format(filename))
        return filename

    # create a copy of audio file and manipulate
    def _mute_wf(self, unmute_span):
        new_wf = unmute_span[0]
        for audio in unmute_span[1:]:
            new_wf = new_wf + self.beep_wf + audio
        return new_wf

    def _speak(self, text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return

    # def _export_speak(self, speak_text, audio_path=None):
    #     if audio_path:
    #         engine = pyttsx3.init()
    #         engine.save_to_file(speak_text, audio_path)
    #         engine.runAndWait()
    #     else:
    #         new_filename = self.audio_path.split(os.sep)[-1]
    #         new_audio_path = self.audio_path[:-len(new_filename)]
    #         ext = new_filename.split(".")[-1]
    #         new_filename = new_filename[:-len(ext) - 1] + "_modified." + ext
    #         engine = pyttsx3.init()
    #         engine.save_to_file(speak_text, os.path.join(new_audio_path, new_filename))
    #         engine.runAndWait()

    def _export_pydub(self, audio_path=None):
        if audio_path:
            self.wf_pydub_modified[self.audio_path].export(audio_path, format="wav")
        else:
            new_filename = self.audio_path.split(os.sep)[-1]
            new_audio_path = self.audio_path[:-len(new_filename)]
            ext = new_filename.split(".")[-1]
            new_filename = new_filename[:-len(ext) - 1] + "_modified." + ext
            self.wf_pydub_modified[self.audio_path].export(os.path.join(new_audio_path, new_filename), format="wav")
