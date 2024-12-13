import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb

model_path = "C:\\Users\\ADMIN\\Desktop\\Llama3_8B_test\\Llama3_8B_test_model"

try:
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)

    if loaded_tokenizer.pad_token is None:
        loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map="auto"
    )

    print("Model loaded successfully (in 8-bit)!")

    batch_size = 1
    prompts = [
        "In the contemporary world, social media has transformed how people interact and maintain relationships. Platforms like Facebook, Instagram, and Twitter have enabled instant communication and forged connections across the globe. However, this digital revolution has also significantly altered the quality and nature of human interactions. One significant impact of social media is the ease with which it facilitates communication. People can now stay in touch with friends and family irrespective of distance, sharing moments and experiences in real-time. This ability to connect instantly has strengthened many long-distance relationships, providing a sense of presence and immediacy that was previously impossible. For instance, video calls and live chats have allowed families separated by oceans to partake in each other's lives more intimately. However, the convenience of social media often comes at the cost of depth and authenticity in interactions. The culture of likes, shares, and comments has fostered an environment where superficial engagement often supersedes meaningful conversation. People may prioritize online validation over genuine connection, leading to shallow and sometimes insincere interactions. Moreover, social media profiles are curated, presenting an idealized version of reality that can distort perceptions and foster unrealistic comparisons. Furthermore, social media has introduced a paradoxical phenomenon where individuals can feel both connected and isolated simultaneously. While these platforms provide the illusion of a large social network, they can exacerbate feelings of loneliness and inadequacy. Studies have shown that excessive social media use is linked to higher rates of depression and anxiety, partly due to the constant comparison with others' seemingly perfect lives. Despite these challenges, social media also offers a platform for positive social change and community building. Movements like #MeToo and Black Lives Matter have gained momentum through social media, highlighting its power to raise awareness and mobilize support for important causes. These platforms enable individuals to share their stories, find solidarity, and collectively advocate for justice and equality. For example, the viral spread of these movements has led to significant societal changes and policy reforms worldwide. In conclusion, social media has undeniably transformed human interaction and relationships, bringing both benefits and drawbacks. While it enhances connectivity and provides a platform for social change, it also poses challenges to the authenticity and depth of human connections. As we navigate this digital landscape, it is crucial to find a balance that leverages the positive aspects of social media while mitigating its potential downsides." +
        "Evaluated this essay based on overall score ,  coherance score, lexical resources , grammartical range and acuraccy and task achiement and give feedback like how an Ielts examinator would do"
    ]

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        inputs = loaded_tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(loaded_model.device)

        with torch.no_grad():
            outputs = loaded_model.generate(
                **inputs,
                max_new_tokens=1200,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                early_stopping=True #Enable early stopping
            )

        for j, output in enumerate(outputs):
            decoded_output = loaded_tokenizer.decode(output, skip_special_tokens=True)
            print(f"Prompt: {batch_prompts[j]}")
            print(f"Response: {decoded_output}")
            print("-" * 20)

except Exception as e:
    print(f"An error occurred: {e}")