import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/LinguifyTTS'.format(ROOT_DIR))
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from src.cli.linguifytts import LinguifyTTS
from src.utils.file_utils import load_wav

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


max_val = 0.8
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )

    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


inference_mode_list = ['Pretrained Voice', '3s Fast Voice Clone', 'Cross-lingual Voice Clone', 'Natural Language Control']
instruct_dict = {
    'Pretrained Voice': '1. Select a pretrained voice\n2. Click the "Generate Audio" button',
    '3s Fast Voice Clone': '1. Select a prompt audio file or record a prompt audio. If both are provided, the audio file takes priority\n2. Enter the prompt text\n3. Click the "Generate Audio" button',
    'Cross-lingual Voice Clone': '1. Select a prompt audio file or record a prompt audio. If both are provided, the audio file takes priority\n2. Click the "Generate Audio" button',
    'Natural Language Control': '1. Enter the instruction text\n2. Click the "Generate Audio" button'
}


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is speech_tts/LinguifyTTS-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if linguifytts.frontend.instruct is False:
            gr.Warning('You are using Natural Language Control mode. The {} model does not support this mode. Please use the speech_tts/LinguifyTTS-300M-Instruct model.'.format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text == '':
            gr.Warning('You are using Natural Language Control mode. Please enter instruct text.')
            return (target_sr, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('You are using Natural Language Control mode. Prompt audio/text will be ignored.')
    # if cross_lingual mode, please make sure that model is speech_tts/LinguifyTTS-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if linguifytts.frontend.instruct is True:
            gr.Warning('You are using Cross-lingual Voice Clone mode. The {} model does not support this mode. Please use the speech_tts/LinguifyTTS-300M model.'.format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('You are using Cross-lingual Voice Clone mode. Instruct text will be ignored.')
        if prompt_wav is None:
            gr.Warning('You are using Cross-lingual Voice Clone mode. Please provide a prompt audio file.')
            return (target_sr, default_data)
        gr.Info('You are using Cross-lingual Voice Clone mode. Make sure the synthesis and prompt texts are in different languages.')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s Fast Voice Clone', 'Cross-lingual Voice Clone']:
        if prompt_wav is None:
            gr.Warning('Prompt audio is missing. Did you forget to provide it?')
            return (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('Prompt audio sample rate {} is below required {}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            return (target_sr, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['Pretrained Voice']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('You are using Pretrained Voice mode. Prompt text/audio and instruct text will be ignored!')
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group == '3s Fast Voice Clone':
        if prompt_text == '':
            gr.Warning('Prompt text is missing. Did you forget to enter it?')
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('You are using 3s Fast Voice Clone mode. Pretrained voice and instruct text will be ignored!')

    if mode_checkbox_group == 'Pretrained Voice':
        logging.info('Received SFT inference request')
        set_all_random_seed(seed)
        output = linguifytts.inference_sft(tts_text, sft_dropdown)
    elif mode_checkbox_group == '3s Fast Voice Clone':
        logging.info('Received zero-shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = linguifytts.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    elif mode_checkbox_group == 'Cross-lingual Voice Clone':
        logging.info('Received cross-lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = linguifytts.inference_cross_lingual(tts_text, prompt_speech_16k)
    else:
        logging.info('Received instruct inference request')
        set_all_random_seed(seed)
        output = linguifytts.inference_instruct(tts_text, sft_dropdown, instruct_text)
    audio_data = output['tts_speech'].numpy().flatten()
    return (target_sr, audio_data)


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### Codebase: [Linguify](https://github.com/josephchay/linguify)")
        gr.Markdown("#### Enter the text to synthesize, choose an inference mode, and follow the instructions.")

        tts_text = gr.Textbox(label="Text to Synthesize", lines=1, value="I am a generative voice model newly launched by the Tongyi Lab Speech Team, offering comfortable and natural speech synthesis.")

        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Select Inference Mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Instructions", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='Select Pretrained Voice', value=sft_spk[0], scale=0.25)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Random Seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Upload Prompt Audio (≥16kHz)')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record Prompt Audio')
        prompt_text = gr.Textbox(label="Prompt Text", lines=1, placeholder="Enter prompt text matching the audio content (auto recognition not supported yet)...", value='')
        instruct_text = gr.Textbox(label="Instruct Text", lines=1, placeholder="Enter instruct text.", value='')

        generate_button = gr.Button("Generate Audio")
        audio_output = gr.Audio(label="Synthesized Audio")

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='speech_tts/Linguify-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    linguifytts = LinguifyTTS(args.model_dir)
    sft_spk = linguifytts.list_avaliable_spks()
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()
