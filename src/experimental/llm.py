from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


prompt_dict = {
    'kimi': [{"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。"},
             {"role": "user", "content": "你好，请注意你现在生成的文字要按照人日常生活的口吻，你的回复将会后续用TTS模型转为语音，并且请把回答控制在100字以内。并且标点符号仅包含逗号和句号，将数字等转为文字回答。"},
             {"role": "assistant", "content": "好的，我现在生成的文字将按照人日常生活的口吻， 并且我会把回答控制在一百字以内, 标点符号仅包含逗号和句号，将阿拉伯数字等转为中文文字回答。下面请开始对话。"}, ],
    'deepseek': [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，请注意你现在生成的文字要按照人日常生活的口吻，你的回复将会后续用TTS模型转为语音，并且请把回答控制在100字以内。并且标点符号仅包含逗号和句号，将数字等转为文字回答。"},
        {"role": "assistant", "content": "好的，我现在生成的文字将按照人日常生活的口吻， 并且我会把回答控制在一百字以内, 标点符号仅包含逗号和句号，将阿拉伯数字等转为中文文字回答。下面请开始对话。"}, ],
    'deepseek_TN': [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，现在我们在处理TTS的文本输入，下面将会给你输入一段文本，请你将其中的阿拉伯数字等等转为文字表达，并且输出的文本里仅包含逗号和句号这两个标点符号"},
        {"role": "assistant", "content": "好的，我现在对TTS的文本输入进行处理。这一般叫做text normalization。下面请输入"},
        {"role": "user", "content": "We paid $123 for this desk."},
        {"role": "assistant", "content": "We paid one hundred and twenty three dollars for this desk."},
        {"role": "user", "content": "详询请拨打010-724654"},
        {"role": "assistant", "content": "详询请拨打零幺零，七二四六五四"},
        {"role": "user", "content": "罗森宣布将于7月24日退市，在华门店超6000家！"},
        {"role": "assistant", "content": "罗森宣布将于七月二十四日退市，在华门店超过六千家。"}, ],
}


class HuggingFaceLLMApi:
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True  # Needed for some models
        )

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

        # Add a final assistant prefix to indicate where the model should generate
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
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode the generated tokens, skipping the input
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return response.strip()
