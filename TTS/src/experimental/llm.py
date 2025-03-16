from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


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
    def __init__(self, model_name="deepseek-ai/deepseek-llm-7b-chat", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model {model_name} on {self.device}...")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use bfloat16 for better performance if available
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

        # Load with appropriate parameters for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=self.device,
                                                          trust_remote_code=True)  # Needed for some models

    def _format_prompt(self, prompt_messages):
        full_prompt = ""

        for message in prompt_messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                full_prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                full_prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                full_prompt += f"<|assistant|>\n{content}\n"

        # Final assistant prefix to indicate where the model should generate
        full_prompt += "<|assistant|>\n"

        return full_prompt

    def call(self, user_question, temperature=0.3, prompt_version='kimi', max_length=512, **kwargs):
        # Get the prompt template and add the user's question
        prompt_messages = prompt_dict[prompt_version] + [{"role": "user", "content": user_question}]

        # Format the prompt for the model
        formatted_prompt = self._format_prompt(prompt_messages)

        # Tokenize the input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=max_length, temperature=temperature, do_sample=temperature > 0, top_p=0.9, pad_token_id=self.tokenizer.eos_token_id, **kwargs)

        # Decode the generated tokens, skipping the input
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return response.strip()
