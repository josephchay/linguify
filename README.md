# Linguify – The Future of Conversational AI

Unlock the power of seamless communication with Linguify, the next-generation call center AI that speaks just like a human - fascinatingly even more. 
Designed with advanced sentiment analysis and deep understanding of human psychology, 
Linguify goes beyond basic chatbot responses. It listens, understands emotions, and responds naturally, 
creating meaningful conversations that build trust and satisfaction.

Whether it's handling complex queries, guiding users with empathy, or delivering personalized experiences, 
Linguify adapts to every situation with unparalleled fluency. 
Empower your business with the most human-like conversational AI, 
and redefine customer interactions with intelligence, care, and precision.

_Linguify – Because Conversations Should Resonate._

# LinguifySpeech TTS System

LinguifySpeech is a Text-to-Speech (TTS) system that can convert written text to natural-sounding speech. 

## Project Setup

To install the LinguifySpeech TTS system, you can use the following commands:

```bash
git clone https://github.com/josephchay/linguify.git
cd linguify
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Basic Usage

To obtain default results, simply run the entire `inference.ipynb` notebook as provided. 
This will execute the necessary steps to generate the expected output.

### Initialize the TTS system

```python
import torch
from src.core import Chat
from IPython.display import Audio

# Initialize the Chat class
chat = Chat()
chat.load_models()  # This will download models from Hugging Face
```

### Generate speech with default settings

> [!NOTE]
> The model is also trained on generic information and can generate speech of equal quality for common topics.

```python
# Prepare your text
texts = [
    # Generic (Mildly Medical)
    ("Welcome to the LinguifySpeech TTS system. "
     "This converts text to natural sounding speech."),
    ("Please ensure you take your medication "
     "as prescribed by your healthcare provider."),
    ("Regular exercise and a balanced diet "
     "are crucial for maintaining heart health."),
    ("In case of a fever, it’s important to stay hydrated "
     "and monitor your temperature."),
    ("If you experience persistent headaches, fatigue, or dizziness, "
     "consult a medical professional for proper evaluation."),

    # Moderately Detailed (Medical Focus)
    ("Patients diagnosed with Type 2 diabetes are advised to monitor "
     "their blood glucose levels regularly, maintain a low glycemic index diet, "
     "and follow their physician's guidance on insulin or oral hypoglycemic medications."),
    ("Post operative care involves wound inspection, dressing changes, "
     "and pain management. Patients should avoid heavy lifting, "
     "keep the incision site dry unless instructed otherwise, "
     "and report signs of infection such as redness, swelling, or discharge."),
    ("Individuals with hypertension should avoid excessive sodium intake, "
     "engage in regular physical activity, and adopt stress reducing techniques "
     "such as meditation or mindfulness exercises. "
     "Monitoring blood pressure at home is recommended to track progress "
     "and ensure stability."),
    ("Asthma management requires identifying triggers, "
     "following a prescribed inhaler routine, "
     "and seeking emergency care if symptoms escalate. "
     "Inhaled corticosteroids and bronchodilators are often key components "
     "in long term management plans."),

    # Highly Detailed (Medical Jargon/Technical)
    ("Beta adrenergic antagonists, commonly known as beta blockers, "
     "function by competitively inhibiting the binding of catecholamines "
     "to beta adrenergic receptors. "
     "This leads to decreased myocardial contractility, reduced heart rate, "
     "and diminished oxygen consumption, making them essential in "
     "the management of ischemic heart disease, chronic heart failure, "
     "and post myocardial infarction care."),
    ("The patient presented with tachycardia, diaphoresis, and altered mentation, "
     "indicative of possible hypoglycemic crisis. "
     "Immediate intervention included administration of 50 percent dextrose "
     "solution intravenously, followed by continuous glucose monitoring "
     "to stabilize serum glucose levels."),
    ("Magnetic resonance imaging, commonly called MRI, revealed a T2 weighted "
     "hyperintense lesion in the left parietal lobe, consistent with gliosis. "
     "The differential diagnosis included post ischemic changes, "
     "demyelinating pathology, or an old vascular insult. "
     "Further correlation with clinical history and CSF analysis was recommended."),

    # Extensive and Detailed (In Depth Medical Descriptions)
    ("Acute coronary syndrome, presents with a spectrum "
     "of conditions ranging from unstable angina to Segment T elevation myocardial infarction. "
     "Management involves prompt initiation of dual antiplatelet therapy "
     "using aspirin and an inhibitor, alongside anticoagulation "
     "with heparin or enoxaparin."),
    ("Sepsis is a life threatening condition characterized by systemic inflammation "
     "and organ dysfunction triggered by infection. "
     "The Surviving Sepsis Campaign recommends initiating broad spectrum antibiotics "
     "within one hour of suspected sepsis, alongside aggressive fluid resuscitation "
     "with crystalloids. "
     "Vasopressor support, particularly norepinephrine, is indicated "
     "if mean arterial pressure, abbreviated as MAP, remains below 65 millimeters "
     "of mercury despite adequate fluid administration."),
]

# Generate speech
wavs = chat.inference(texts, use_decoder=True)

# Play the audio (if in a notebook environment)
from IPython.display import Audio
Audio(wavs[0], rate=24_000)

# Or save the audio to a file
import scipy.io.wavfile as wav
wav.write("output.wav", 24000, wavs[0])
```

## Advanced Usage: Custom Speaker Voice

You can customize the voice characteristics by using a speaker embedding:

```python
# Import necessary libraries
from huggingface_hub import snapshot_download
import os

# Download speaker statistics
download_path = snapshot_download(repo_id='josephchay/LinguifySpeech', allow_patterns=["*.pt"])
spk_stat = torch.load(os.path.join(download_path, 'asset', 'spk_stat.pt'))

# Generate a random speaker embedding based on the statistics
rand_spk = torch.randn(768) * spk_stat.chunk(2)[0] + spk_stat.chunk(2)[1]

# Set inference parameters
params_infer_code = {'spk_emb': rand_spk, 'temperature': 0.3}
params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}

# Generate speech with custom parameters
wav = chat.inference(
    text='This is a demonstration of speech synthesis with a custom voice.',
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
    use_decoder=True
)

# Play or save the audio
Audio(wav[0], rate=24_000)
```

## Parameters

- `use_decoder=True`: Uses a higher quality decoder for better results
- `skip_refine_text=True`: Skips the LLM-based text refinement step
- `params_infer_code`: Dictionary of parameters for the inference process
  - `spk_emb`: Speaker embedding for voice characteristics
  - `temperature`: Controls randomness in generation (lower = more deterministic)
- `params_refine_text`: Dictionary of parameters for text refinement
  - `prompt`: Special tags that can modify speech style

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

## Updates

Refer to the [CHANGELOG](CHANGELOG.md) file for the thorough latest updates and features.
