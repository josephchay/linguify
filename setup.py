from setuptools import setup, find_packages
import os


readme_path = "README.md"
with open(readme_path, encoding="utf-8") as f:
    long_description = f.read()


def get_requirements(filename):
    requirements = []
    if os.path.exists(filename):
        with open(filename, encoding="utf-8") as f:
            requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]
    return requirements


requirements = []
tts_requirements = get_requirements("TTS/requirements.txt")
llm_requirements = get_requirements("LLM/requirements.txt")
all_requirements = list(set(tts_requirements + llm_requirements))


setup(
    name="linguify",
    version="0.0.0",
    packages=find_packages(),
    install_requires=all_requirements,
    description="Because conversations should resonate.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/josephchay/linguify",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence :: Machine Learning",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Programming Language :: Python :: 3.12.7",
    ],
    keywords="tts, text-to-speech, speech synthesis, llm, nlp",
    python_requires=">=3.12.7",
    include_package_data=True,
    package_data={
        "": ["TTS/models/*", "LLM/models/*"],
    },
    extras_require={
        "tts": tts_requirements,
        "llm": llm_requirements,
        "all": all_requirements,
    },
)
