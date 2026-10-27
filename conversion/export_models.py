"""Export local Uzbek weights. Run from project root: python -m conversion.export_models STAGE."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from safetensors.torch import load_file
from .vendor.chatterbox.models.t3 import T3
from .vendor.chatterbox.models.t3.modules.t3_config import T3Config
from .wrappers import Embeddings, LlamaForCFG, Decoder

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
OUT = ROOT / "models"
torch.set_num_threads(4)

def load_t3():
    model = T3(T3Config.multilingual()).eval()
    model.load_state_dict(load_file(CKPT / "t3.safetensors"), strict=True)
    model.requires_grad_(False)
    return model

def load_s3():
    from .vendor.chatterbox.models.s3gen import S3Gen
    model = S3Gen().eval()
    missing, unexpected = model.load_state_dict(load_file(CKPT / "s3gen.safetensors"), strict=False)
    allowed = {"tokenizer._mel_filters", "tokenizer.window"}
    if set(missing) - allowed or unexpected:
        raise RuntimeError((missing, unexpected))
    return model.requires_grad_(False)

def save_tensor(name, array):
    array=np.asarray(array)
    dtype="int64" if array.dtype.kind in "iu" else "float32"
    array=array.astype("<i8" if dtype=="int64" else "<f4")
    filename=name + ".bin"
    array.tofile(OUT / filename)
    return {"file":filename,"dtype":dtype,"shape":list(array.shape)}

def prepare():
    from .vendor.chatterbox.tts import ChatterboxTTS, punc_norm
    from .vendor.chatterbox.models.voice_encoder import VoiceEncoder
    from .vendor.chatterbox.models.tokenizers import EnTokenizer
    t3=load_t3(); s3=load_s3(); ve=VoiceEncoder().eval()
    ve.load_state_dict(load_file(CKPT / "ve.safetensors"))
    ve.requires_grad_(False)
    engine=ChatterboxTTS(t3,s3,ve,EnTokenizer(str(CKPT/"tokenizer.json")),"cpu")
    with torch.no_grad():
        engine.prepare_conditionals(str(CKPT/"reference_voice.wav"))
        cond=t3.prepare_conditioning(engine.conds.t3)
    tensors={
        "conditioning": cond[:,:-1].numpy(),
        "prompt_tokens": engine.conds.gen["prompt_token"].numpy(),
        "speaker_embeddings": engine.conds.gen["embedding"].numpy(),
        "speaker_features": engine.conds.gen["prompt_feat"].numpy(),
    }
    manifest={"format_version":1,"model":"uzbek-chatterbox","sample_rate":24000,
        "voice":"default","voice_cloning":False,"dtype":"float32","num_layers":30,
        "num_heads":16,"head_dim":64,"hidden_size":1024,"start_text_token":255,
        "stop_text_token":0,"start_speech_token":6561,"stop_speech_token":6562,
        "speech_vocab_size":6561,"max_new_tokens":1000,"cfm_steps":10,"harmonics":9,
        "samples_per_token":960,"graphs":{"embeddings":"embeddings.onnx",
        "language_model":"language_model.onnx","decoder":"decoder.onnx"},
        "tensors":{k:save_tensor(k,v) for k,v in tensors.items()}}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    shutil.copy2(CKPT/"tokenizer.json",OUT/"tokenizer.json")
    text=punc_norm("Salom, yaxshimisiz?")
    ids=torch.tensor([[255]+engine.tokenizer.encode(text)+[0]],dtype=torch.long).repeat(2,1)
    with torch.no_grad():
        embeds,_=t3.prepare_input_embeds(t3_cond=engine.conds.t3,text_tokens=ids,
            speech_tokens=torch.full((2,1),6561,dtype=torch.long),cfg_weight=.5)
        bos=t3.speech_emb(torch.full((2,1),6561,dtype=torch.long))+t3.speech_pos_emb.get_fixed_embedding(0)
        embeds=torch.cat([embeds,bos],dim=1)
    np.savez(ROOT/"conversion/prefill_fixture.npz",ids=ids.numpy(),embeds=embeds.numpy())
    print("Voice tensor shapes:", {k:list(v.shape) for k,v in tensors.items()},flush=True)

def export(module,args,path,names,outputs,axes):
    module.eval().requires_grad_(False)
    print("Exporting",path.name,flush=True)
    started=time.time()
    with torch.no_grad():
        torch.onnx.export(module,args,str(path),opset_version=17,input_names=names,
            output_names=outputs,dynamic_axes=axes,do_constant_folding=True,
            dynamo=False,external_data=True)
    print("Exported",path.name,"in",round(time.time()-started,1),"seconds",flush=True)

def ort_session(path):
    import onnxruntime as ort
    options=ort.SessionOptions(); options.intra_op_num_threads=4
    options.inter_op_num_threads=1
    return ort.InferenceSession(str(path),options,providers=["CPUExecutionProvider"])

def embeddings():
    model=Embeddings(load_t3()).eval()
    fixture=np.load(ROOT/"conversion/prefill_fixture.npz")
    text=fixture["ids"]
    ids=np.concatenate([text,np.full((2,2),6561,dtype=np.int64)],axis=1)
    pos=np.concatenate([np.tile(np.arange(text.shape[1]),(2,1)),np.zeros((2,2),np.int64)],axis=1)
    kinds=np.concatenate([np.zeros_like(text),np.ones((2,2),np.int64)],axis=1)
    values=(torch.from_numpy(ids),torch.from_numpy(pos),torch.from_numpy(kinds),torch.tensor(.5))
    export(model,values,OUT/"embeddings.onnx",["input_ids","position_ids","token_kind","exaggeration"],
        ["inputs_embeds","emotion_embed"],{k:{1:"sequence_length"} for k in ("input_ids","position_ids","token_kind","inputs_embeds")})
    session=ort_session(OUT/"embeddings.onnx")
    result,emotion=session.run(None,dict(zip(["input_ids","position_ids","token_kind","exaggeration"],
        [ids,pos,kinds,np.array(.5,np.float32)])))
    prefix=np.fromfile(OUT/"conditioning.bin",dtype="<f4").reshape(1,-1,1024)
    actual=np.concatenate([np.repeat(prefix,2,axis=0),np.repeat(emotion,2,axis=0),result],axis=1)
    np.testing.assert_allclose(actual,fixture["embeds"],rtol=1e-5,atol=1e-6)
    print("Embeddings match the original PyTorch prefill",flush=True)

def language_model():
    model=LlamaForCFG(load_t3()).eval()
    names=["inputs_embeds","attention_mask","cfg_weight"]
    outputs=["logits","attn_layers"]
    axes={"inputs_embeds":{1:"sequence_length"},"attention_mask":{1:"total_sequence_length"},
        "logits":{1:"sequence_length"},"attn_layers":{2:"sequence_length",3:"total_sequence_length"}}
    for i in range(30):
        for kv in ("key","value"):
            key=f"past_key_values.{i}.{kv}"; name=f"present.{i}.{kv}"
            names.append(key);outputs.append(name)
            axes[key]={2:"past_sequence_length"};axes[name]={2:"total_sequence_length"}
    torch.manual_seed(17)
    args=(torch.randn(2,3,1024),torch.ones(2,8,dtype=torch.long),torch.tensor(.5),
        *[torch.randn(2,16,5,64)*.01 for _ in range(60)])
    export(model,args,OUT/"language_model.onnx",names,outputs,axes)
    # Save reference results before freeing PyTorch weights to keep peak memory bounded.
    fixture=np.load(ROOT/"conversion/prefill_fixture.npz")
    reference=[]
    for q,p in ((2,0),(1,7)):
        values=(torch.randn(2,q,1024),torch.ones(2,p+q,dtype=torch.long),torch.tensor(.5),
            *[torch.randn(2,16,p,64)*.01 for _ in range(60)])
        with torch.no_grad(): expected=[v.numpy().copy() for v in model(*values)]
        reference.append(([v.numpy().copy() for v in values],expected))
    del model,args;gc.collect()
    session=ort_session(OUT/"language_model.onnx")
    errors=[]
    for values,expected in reference:
        actual=session.run(None,dict(zip(names,values)))
        for a,b in zip(actual,expected):
            np.testing.assert_allclose(a,b,rtol=5e-3,atol=8e-4)
        errors.append(float(np.max(np.abs(actual[0]-expected[0]))))
    (ROOT/"docs/validation-language-model.json").write_text(json.dumps({"logits_max_abs_errors":errors,"zero_past_and_cached_decode":"passed"},indent=2))
    print("LM parity passed, max logits errors",errors,flush=True)

def decoder():
    model=Decoder(load_s3(),n_cfm_timesteps=10).eval()
    cfg=json.loads((OUT/"manifest.json").read_text())
    def read(name):
        spec=cfg["tensors"][name]
        return torch.from_numpy(np.fromfile(OUT/spec["file"],dtype="<i8" if spec["dtype"]=="int64" else "<f4").reshape(spec["shape"]))
    prompt=read("prompt_tokens");emb=read("speaker_embeddings");feat=read("speaker_features")
    torch.manual_seed(17)
    token=torch.cat([prompt,torch.randint(0,6561,(1,30))],dim=1)
    flow=torch.randn(1,80,token.shape[1]*2)
    noise=torch.randn(1,9,(token.shape[1]*2-feat.shape[1])*480)
    args=(token,emb,feat,flow,noise)
    with torch.no_grad(): expected=model(*args).numpy().copy()
    np.savez(ROOT/"conversion/decoder_fixture.npz",tokens=token.numpy(),embedding=emb.numpy(),
        features=feat.numpy(),flow_noise=flow.numpy(),source_noise=noise.numpy(),waveform=expected)
    names=["speech_tokens","speaker_embeddings","speaker_features","flow_noise","source_noise"]
    export(model,args,OUT/"decoder.onnx",names,["waveform"],{
        "speech_tokens":{1:"total_tokens"},"speaker_features":{1:"prompt_frames"},
        "flow_noise":{2:"total_frames"},"source_noise":{2:"samples"},"waveform":{1:"samples"}})
    del model,args;gc.collect()
    session=ort_session(OUT/"decoder.onnx")
    fixture=np.load(ROOT/"conversion/decoder_fixture.npz")
    actual=session.run(None,dict(zip(names,[fixture[k] for k in ("tokens","embedding","features","flow_noise","source_noise")])))[0]
    expected=fixture["waveform"]
    cosine=float(np.dot(actual.ravel(),expected.ravel())/(np.linalg.norm(actual)*np.linalg.norm(expected)))
    assert cosine > .98, cosine
    assert np.isfinite(actual).all()
    (ROOT/"docs/validation-decoder.json").write_text(json.dumps({"waveform_cosine":cosine,"max_abs_error":float(np.max(np.abs(actual-expected)))},indent=2))
    print("Decoder waveform parity cosine",cosine,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["prepare","embeddings","language_model","decoder"])
    stage=parser.parse_args().stage
    globals()[stage]()
