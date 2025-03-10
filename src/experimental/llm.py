from openai import OpenAI


prompt_dict = {
    'deepseek': [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Please note that you should respond in a natural, conversational tone. Your responses will be converted to speech using a TTS model, so please keep your answers under 100 words. Use only commas and periods for punctuation, and spell out all numbers in word form."},
        {"role": "assistant", "content": "I understand. I'll respond in a natural, conversational tone and keep my answers under one hundred words. I'll only use commas and periods for punctuation, and I'll convert all numbers to words. Let's begin our conversation."},
    ],
    'deepseek_TN': [
        {"role": "system", "content": "You are a helpful assistant specialized in text normalization for TTS systems."},
        {"role": "user", "content": "We're processing text input for a TTS system. I'll give you some text, and I want you to convert all numbers, symbols, and special characters into their spoken word form. Use only commas and periods for punctuation in your output."},
        {"role": "assistant", "content": "I understand. I'll normalize the text for TTS processing, often called text normalization. Please provide the text."},
        {"role": "user", "content": "We paid $123 for this desk."},
        {"role": "assistant", "content": "We paid one hundred and twenty three dollars for this desk."},
        {"role": "user", "content": "Call 555-123-4567 for more info!"},
        {"role": "assistant", "content": "Call five five five, one two three, four five six seven for more info."},
        {"role": "user", "content": "The meeting is on 7/24 at 3:30PM with 6000 attendees."},
        {"role": "assistant", "content": "The meeting is on July twenty fourth at three thirty PM with six thousand attendees."},
    ],
}


class LLMApi:
    def __init__(self, api_key, base_url, model):
        self.client =  OpenAI(
            api_key = api_key,
            base_url = base_url,
        )
        self.model = model

    def call(self, user_question, temperature = 0.3, prompt_version='kimi', **kwargs):
    
        completion = self.client.chat.completions.create(
            model = self.model,
            messages = prompt_dict[prompt_version]+[{"role": "user", "content": user_question},],
            temperature = temperature,
            **kwargs
        )
        return completion.choices[0].message.content
