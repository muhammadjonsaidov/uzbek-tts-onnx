"""Export adapters; model weights are unchanged. See THIRD_PARTY.md."""
import torch
from torch import nn
_old_item = torch.Tensor.item
from .vendor.onnx_recipe.chatterbox.export_language_model import LlamaForCFG
from .vendor.onnx_recipe.chatterbox.export_conditional_decoder import ConditionalDecoder
# The upstream constants module globally patches Tensor.item. Do not retain that patch.
torch.Tensor.item = _old_item

class Embeddings(nn.Module):
    def __init__(self, t3):
        super().__init__()
        self.text = t3.text_emb
        self.text_pos = t3.text_pos_emb.emb
        self.speech = t3.speech_emb
        self.speech_pos = t3.speech_pos_emb.emb
        self.emotion = t3.cond_enc.emotion_adv_fc
    def forward(self, input_ids, position_ids, token_kind, exaggeration):
        is_speech = token_kind.bool()
        text_ids = torch.where(is_speech, torch.zeros_like(input_ids), input_ids)
        speech_ids = torch.where(is_speech, input_ids, torch.zeros_like(input_ids))
        text_pos = torch.where(is_speech, torch.zeros_like(position_ids), position_ids)
        speech_pos = torch.where(is_speech, position_ids, torch.zeros_like(position_ids))
        rows = torch.arange(input_ids.shape[0], device=input_ids.device).view(-1, 1, 1)
        # Upstream CFG clears token embeddings BEFORE adding positional embeddings.
        text = self.text(text_ids) * (rows != 1).to(torch.float32) + self.text_pos(text_pos)
        speech = self.speech(speech_ids) + self.speech_pos(speech_pos)
        embedded = torch.where(is_speech.unsqueeze(-1), speech, text)
        return embedded, self.emotion(exaggeration.view(1, 1, 1))

class Decoder(ConditionalDecoder):
    def forward(self, speech_tokens, speaker_embeddings, speaker_features, flow_noise, source_noise):
        self.flow_noise = flow_noise
        self.source_noise = source_noise
        return super().forward(speech_tokens, speaker_embeddings, speaker_features)
