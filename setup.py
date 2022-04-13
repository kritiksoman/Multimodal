from setuptools import setup, find_packages
import os

here = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(here, "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name="multimodal",  # Required
    version="0.0.1",  # Required
    description="Multimodal ",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/kritiksoman/Multimodal",  # Optional
    author="Kritik Soman",  # Optional
    author_email="kritiksoman2020@iitkalumni.org",  # Optional
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3.8",
        # 'Programming Language :: Python :: 2.7 :: Only',
    ],
    keywords="sample, setuptools, development",  # Optional
    packages=find_packages(),
    python_requires=">=2.7",
    include_package_data=True,  # to include manifest.in
    install_requires=[
        "patool",
        'pyunpack',
        "vosk",
        "pydub",
        "youtube_dl",
        'torch',
        "transformers",
        "numpy",
        "pandas",
        "pdfminer.six",
        "pyttsx3==2.7",
        "python-docx",
        "gdown"
    ]
)