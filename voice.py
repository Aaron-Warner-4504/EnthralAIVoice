import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "ai4bharat/indic-parler-tts"

model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

desc_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

# -----------------------
# CHANGE THIS FOR HINDI / INDIAN ENGLISH / MALE / FEMALE
# -----------------------
description = """
Hindi female voice, expressive, clear Indian accent, native Hindi prosody, natural pauses, and smooth pronunciation of Hindi phonetics.

"""

# TEXT YOU WANT TO SPEAK
prompt = "नमस्ते! आप कैसे हैं? मैं आपकी सहायता के लिए यहाँ हूँ।"

# -----------------------
# Generate voice
# -----------------------
desc_inputs = desc_tokenizer(description, return_tensors="pt").to(device)
prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)

audio = model.generate(
    input_ids=desc_inputs.input_ids,
    attention_mask=desc_inputs.attention_mask,
    prompt_input_ids=prompt_inputs.input_ids,
    prompt_attention_mask=prompt_inputs.attention_mask,
    temperature=0.55,         # smoother tone
    repetition_penalty=1.0,  # natural flow
)

audio_np = audio.cpu().numpy().squeeze()
sf.write("hindi_voice8.wav", audio_np, model.config.sampling_rate)
