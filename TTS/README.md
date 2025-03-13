# LinguifySpeech TTS Agent Architecture System

LinguifySpeech is a Text-to-Speech (TTS) system that can convert written text to natural-sounding speech. 

For our agent model assets and config files, refer to [Hugging Face model hub](https://huggingface.co/josephchay/linguifySpeech).

## Package Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/josephchay/linguify.git
```

```bash
# Or for TTS component only
pip install "git+https://github.com/josephchay/linguify.git#egg=linguify[tts]"
```

## Development

### Clone the Repository

To install the LinguifySpeech TTS system, you can use the following commands:

```bash
git clone https://github.com/josephchay/linguify.git
```

### Install Dependencies

```bash
cd linguify
pip install -e .
```

or for manual installation without `setup.py`:

```bash
cd linguify/TTS
pip install -r requirements.txt
```

## Basic Usage

To obtain default results, simply run the entire `inference.ipynb` notebook as provided. 
This will execute the necessary steps to generate the expected output.

For extended inference techniques with more detailed control and customization, 
refer to the `extended_inference.ipynb` notebook.

### Initialize the TTS system

```python
import torch

from src.core import Chat

chat = Chat()
chat.load_models(compile=False)  # Set to True for better performance
```

### Text Prompts

> [!NOTE]
> The model is also trained on generic information and can generate speech of equal quality for common topics.
> 
> _Regarding textual abbreviation, each of them should be presented with spaced-out characters to ensure clarity and accurate interpretation._

```python
# Prepare your text
texts = [
  "<YOUR TEXT PROMPT HERE>",
  "<YOUR TEXT PROMPT HERE>",
  "<YOUR TEXT PROMPT HERE>",
]
```

### Generate Speech

```python
# Generate speech
wavs = chat.inference(texts, use_decoder=True)

torchaudio.save("audio_output1.wav", torch.from_numpy(wavs[0]), 24000)

# Play the audio (if in a notebook environment)
from IPython.display import Audio
Audio(wavs[0], rate=24_000)
```

## Advanced Usage

```python
# Sample a speaker from Gaussian.
rand_spk = chat.sample_random_speaker()
params_infer_code = {
  'spk_emb': rand_spk, # add sampled speaker 
  'temperature': .3, # using custom temperature
  'top_P': 0.7, # top P decode
  'top_K': 20, # top K decode
}

# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7) 
# to generate special token in text to synthesize.
params_refine_text = {
  'prompt': '[oral_2][laugh_0][break_6]'
} 

wav = chat.inference(texts, params_refine_text=params_refine_text, params_infer_code=params_infer_code)

# For word level manual control.
text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
wav = chat.inference(text, skip_refine_text=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
torchaudio.save("audio_output2.wav", torch.from_numpy(wavs[0]), 24000)
```

## Example Usage

```python
inputs_en = """
Patients presenting with [uv_break]tachyarrhythmia[uv_break] may require immediate intervention, particularly in 
cases of [uv_break]ventricular fibrillation[laugh] oh, not ideal timing for that[laugh] [uv_break]where 
synchronized cardioversion is indicated. [uv_break]Beta blockers like [uv_break]metoprolol[uv_break] or [uv_break]propranolol[uv_break] may 
be administered, but [uv_break]contraindications such as [uv_break]bradycardia[uv_break] or [uv_break]severe asthma[uv_break] must be carefully assessed. 
[laugh]You wouldn't want to mix those up![laugh] [uv_break]In critical scenarios, [uv_break]amiodarone[uv_break] or [uv_break]lidocaine[uv_break] 
may be considered, ensuring [uv_break]continuous E C G monitoring[uv_break] to evaluate Q T interval prolongation risks. [uv_break]Ultimately,[uv_break] 
clinical judgment is paramount, [uv_break]so tread carefully when managing complex arrhythmias.[uv_break]
""".replace('\n', '') # English is still experimental.

params_refine_text = {
  'prompt': '[oral_2][laugh_0][break_4]'
}
# audio_array_cn = chat.inference(inputs_cn, params_refine_text=params_refine_text)
audio_array_en = chat.inference(inputs_en, params_refine_text=params_refine_text)
torchaudio.save("audio_output3.wav", torch.from_numpy(audio_array_en[0]), 24000)
```

## Sample Audio Outputs

Prompt:

"Pharmacokinetic analysis indicates that the prescribed medication reaches peak plasma concentration within two hours, 
with a half-life of approximately eight hours, necessitating bi-daily administration."

Output:

https://github.com/user-attachments/assets/b75f3872-6861-45dd-a109-39394b0f72d9

Prompt:

"Medical advancements in AI-driven diagnostics have significantly improved early cancer detection. 
Techniques such as liquid biopsy and deep-learning-based imaging analysis enhance accuracy and patient outcomes."

Output:

https://github.com/user-attachments/assets/2cbeb712-a20a-47ee-957b-7b142652fa7d

The audio outputs above not only demonstrate natural-sounding speech but also showcase the system's ability to 
handle complex medical terminology with ease, as well as pronouncing them in a very rich tone, with strong pause-like smoothing slurs.

## Features & Updates

Refer to the [CHANGELOG](CHANGELOG.md) file for the thorough latest updates and features of **LinguifySpeech TTS**.
