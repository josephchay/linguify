import os
import random
import argparse

import torch
import gradio as gr
import numpy as np

import src


# Voice Options: Preset configurations for suitable voice tones
voices = {
    "默认": {"seed": 2},
    "音色1": {"seed": 1111},
    "音色2": {"seed": 2222},
    "音色3": {"seed": 3333},
    "音色4": {"seed": 4444},
    "音色5": {"seed": 5555},
    "音色6": {"seed": 6666},
    "音色7": {"seed": 7777},
    "音色8": {"seed": 8888},
    "音色9": {"seed": 9999},
    "音色10": {"seed": 11111},
}


def generate_seed():
    new_seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": new_seed
    }


def on_voice_change(vocie_selection):
    return voices.get(vocie_selection)['seed']


def generate_audio(text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag):
    torch.manual_seed(audio_seed_input)
    rand_spk = chat.sample_random_speaker()
    params_infer_code = {
        'spk_emb': rand_spk,
        'temperature': temperature,
        'top_P': top_P,
        'top_K': top_K,
    }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}

    torch.manual_seed(text_seed_input)

    if refine_text_flag:
        text = chat.inference(text, skip_refine_text=False, refine_text_only=True,
                              params_refine_text=params_refine_text, params_infer_code=params_infer_code)
    wav = chat.inference(text, skip_refine_text=True, params_refine_text=params_refine_text,
                         params_infer_code=params_infer_code)

    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000
    text_data = text[0] if isinstance(text, list) else text

    return [(sample_rate, audio_data), text_data]


def generate_audio_stream(text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag):
    torch.manual_seed(audio_seed_input)
    rand_spk = chat.sample_random_speaker()
    params_infer_code = {
        'spk_emb': rand_spk,
        'temperature': temperature,
        'top_P': top_P,
        'top_K': top_K,
    }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}

    torch.manual_seed(text_seed_input)

    wavs_gen = chat.inference(text, skip_refine_text=True, params_refine_text=params_refine_text,
                              params_infer_code=params_infer_code, stream=True)

    for gen in wavs_gen:
        wavs = [np.array([[]])]
        wavs[0] = np.hstack([wavs[0], np.array(gen[0])])
        audio = wavs[0][0]

        max_audio = np.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio /= max_audio

        yield 24000, (audio * 32768).astype(np.int16)


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# LinguifySpeech Web UI")
        gr.Markdown("LinguifySpeech Model Assets and Config Files: [josephchay/LinguifySpeech](https://github.com/josephchay/LinguifySpeech)")

        default_text = ("The patient presents with acute respiratory distress syndrome (A R D S), "
                        "characterized by bilateral diffuse alveolar infiltrates and a PaO2/FiO2 ratio below 200 mmHg, "
                        "indicating severe oxygenation impairment. Immediate initiation of mechanical ventilation and "
                        "close monitoring of arterial blood gas analysis are recommended to optimize ventilatory parameters.")
        text_input = gr.Textbox(label="Input Text", lines=4, placeholder="Please Input Text...", value=default_text)

        with gr.Row():
            refine_text_checkbox = gr.Checkbox(label="Refine text", value=True)
            temperature_slider = gr.Slider(minimum=0.00001, maximum=1.0, step=0.00001, value=0.3, label="Audio temperature")
            top_p_slider = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.7, label="top_P")
            top_k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=20, label="top_K")

        with gr.Row():
            voice_options = {}
            voice_selection = gr.Dropdown(label="Voice", choices=voices.keys(), value='Default')
            audio_seed_input = gr.Number(value=2, label="Audio Seed")
            generate_audio_seed = gr.Button("\U0001F3B2")
            text_seed_input = gr.Number(value=42, label="Text Seed")
            generate_text_seed = gr.Button("\U0001F3B2")

        generate_button = gr.Button("Generate")
        stream_generate_button = gr.Button("Streaming Generate")

        text_output = gr.Textbox(label="Output Text", interactive=False)
        audio_output = gr.Audio(label="Output Audio", value=None, streaming=True, autoplay=True, interactive=False,
                                show_label=True)

        # 使用Gradio的回调功能来更新数值输入框
        voice_selection.change(fn=on_voice_change, inputs=voice_selection, outputs=audio_seed_input)

        generate_audio_seed.click(generate_seed,
                                  inputs=[],
                                  outputs=audio_seed_input)

        generate_text_seed.click(generate_seed,
                                 inputs=[],
                                 outputs=text_seed_input)

        generate_button.click(generate_audio,
                              inputs=[text_input, temperature_slider, top_p_slider, top_k_slider, audio_seed_input,
                                      text_seed_input, refine_text_checkbox],
                              outputs=[audio_output, text_output])

        stream_generate_button.click(generate_audio_stream,
                                     inputs=[text_input, temperature_slider, top_p_slider, top_k_slider,
                                             audio_seed_input, text_seed_input, refine_text_checkbox],
                                     outputs=[audio_output])

        gr.Examples(
            examples=[
                ["In cases of septic shock with multi-organ dysfunction syndrome (M O D S), patients often exhibit profound vasodilation, "
                 "capillary leak syndrome, and impaired cellular oxygen utilization. Management involves aggressive fluid resuscitation, "
                 "vasopressor support (commonly norepinephrine), and targeted antimicrobial therapy. Continuous renal replacement therapy (C R R T) "
                 "may be indicated in cases of severe acute kidney injury to manage electrolyte imbalances and fluid overload.",
                 0.3, 0.7, 20, 2, 42, True],
                ["Administering 0.9% saline [uv_break] is standard for fluid resuscitation, just don't mistake it for your morning coffee![laugh][lbreak]",
                 0.5, 0.5, 10, 245, 531, True],
                ["Patients presenting with [uv_break] acute myocardial infarction often describe a crushing chest pain "
                 "radiating to the left arm or jaw. [uv_break] Immediate administration of antiplatelet therapy, such as aspirin, "
                 "along with oxygen, nitrates, and morphine, is crucial in the acute phase. [uv_break] Beta-blockers may be "
                 "introduced to reduce myocardial oxygen demand, while reperfusion strategies — either percutaneous coronary intervention (P C I) "
                 "or thrombolytic therapy — are essential for restoring blood flow. [laugh] Managing comorbidities like hypertension and "
                 "diabetes is vital for long-term recovery.",
                 0.2, 0.6, 15, 67, 165, True],
            ],
            inputs=[text_input, temperature_slider, top_p_slider, top_k_slider, audio_seed_input, text_seed_input, refine_text_checkbox],
        )

    parser = argparse.ArgumentParser(description='LinguifySpeech Demo')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
    parser.add_argument('--server_port', type=int, default=8080, help='Server port')
    parser.add_argument('--root_path', type=str, default=None, help='Root Path')
    parser.add_argument('--custom_path', type=str, default=None, help='the custom model path')
    args = parser.parse_args()

    global chat
    chat = src.Chat()

    if args.custom_path == None:
        chat.load_models()
    else:
        print('local model path:', args.custom_path)
        chat.load_models('local', custom_path=args.custom_path)

    demo.launch(server_name=args.server_name, server_port=args.server_port, inbrowser=True)


if __name__ == '__main__':
    main()
