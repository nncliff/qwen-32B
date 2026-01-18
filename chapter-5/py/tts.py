import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random

def text_preprocess(text):
    normalized = " ".join(text.strip().split()).lower()
    # Simple phoneme conversion: for demonstration, we just split by characters
    phoneme_seq = [word[0] for word in normalized.split()]
    return phoneme_seq

class MelSpectrogramGenerator(nn.Module):
    def __init__(self, in_dim, mel_dim):
        super(MelSpectrogramGenerator, self).__init__()
        self.fc1 = nn.Linear(in_dim, mel_dim)

    def forward(self, x):
        mel = self.fc1(x)  # (batch_size, mel_dim)
        mel = F.relu(mel)
        return mel  # (batch_size, mel_dim)

class DummyVocoder(nn.Module):
    def __init__(self, mel_dim, audio_dim):
        super(DummyVocoder, self).__init__()
        self.fc1 = nn.Linear(mel_dim, audio_dim)

    def forward(self, mel):
        pooled = torch.mean(mel, dim=1)  # (batch_size, mel_dim)
        audio = self.fc1(pooled)  # (batch_size, audio_dim)
        audio = torch.tanh(audio)
        return audio  # (batch_size, audio_dim)

class SimpleTTSModel(nn.Module):
    def __init__(self, phoneme_vocab_size=50, phoneme_embed_dim=256, mel_dim=80, audio_dim=16000):
        super(SimpleTTSModel, self).__init__()
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, phoneme_embed_dim)
        self.mel_generator = MelSpectrogramGenerator(phoneme_embed_dim, mel_dim)
        self.vocoder = DummyVocoder(mel_dim, audio_dim)

    def forward(self, text_embeddings):
        phoneme_embeds = self.phoneme_embedding(text_embeddings)  # (batch_size, seq_length, phoneme_embed_dim)
        mel_spectrograms = self.mel_generator(phoneme_embeds)    # (batch_size, mel_dim)
        audio_waveforms = self.vocoder(mel_spectrograms)         # (batch_size, audio_dim)
        return audio_waveforms  # (batch_size, mel_dim), (batch_size, audio_dim)

def phoneme_to_indices(phoneme_seq, phoneme_to_idx):
    return [phoneme_to_idx.get(p, 0) for p in phoneme_seq]

def simulate_tts():
    input_texts = "Hello world! This is a test of the TTS system."
    print("Input Text:", input_texts)

    phoneme_seq = text_preprocess(input_texts)
    print("Phoneme Sequence:", phoneme_seq)

    phoneme_to_idx = {chr(i + 97): i + 1 for i in range(26)}  # a-z mapped to 1-26
    phoneme_vocab_size = 27 # including unknown token 0
    phoneme_indices = phoneme_to_indices(phoneme_seq, phoneme_to_idx)
    print("Phoneme Indices:", phoneme_indices)

    # Convert to tensor
    # phoneme_tensor = torch.tensor(phoneme_indices, dtype=torch.long).unsqueeze(0)  # (1, seq_length)
    phoneme_tensor = torch.tensor([phoneme_indices], dtype=torch.long)  # (1, seq_length)

    phoneme_embedd_dim = 64
    mel_dim = 128
    audio_dim = 16000 # 1 second of audio at 16kHz

    tts_model = SimpleTTSModel(phoneme_vocab_size, phoneme_embedd_dim, mel_dim, audio_dim)
    # .eval() does NOT disable gradient computation. For inference, you typically combine it with torch.no_grad():
    tts_model.eval() # Set model to evaluation mode, 
    with torch.no_grad():
        audio_output = tts_model(phoneme_tensor)  # (1, audio_dim)

    waveform_np = audio_output.cpu().numpy()  # (1, audio_dim)
    print("Generated Audio Waveform Shape:", waveform_np.shape)
    print("Generated Audio Waveform (first 10 samples):", waveform_np[0][:10])

if __name__ == "__main__":
    simulate_tts()