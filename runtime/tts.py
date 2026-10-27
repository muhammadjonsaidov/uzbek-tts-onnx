"""Offline ONNX inference. Imports no torch, transformers, or Chatterbox source."""
from pathlib import Path
import argparse
import json
import re
import time
import wave
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from .alignment import AlignmentStreamAnalyzer


def normalize(text):
    text=text.strip()
    if not text: raise ValueError("Text is empty")
    if text[0].islower(): text=text[0].upper()+text[1:]
    text=" ".join(text.split())
    for old,new in [("...",", "),("…",", "),(":",","),(" - ",", "),(";",","),("—","-"),("–","-"),(" ,",","),("“",'"'),("”",'"'),("‘", "'"),("’", "'")]:
        text=text.replace(old,new)
    if not text.endswith((".","!","?","-",",")): text+="."
    return text


def split_text(text,limit=140):
    words=re.findall(r"\S+",text)
    chunks=[];current=""
    for word in words:
        for start in range(0,len(word),limit):
            part=word[start:start+limit]
            candidate=(current+" "+part).strip()
            if len(candidate)>limit:
                chunks.append(current);current=part
            else: current=candidate
    if current:chunks.append(current)
    if not chunks:raise ValueError("Text is empty")
    return chunks


class UzbekTTS:
    def __init__(self, model_dir, threads=4):
        self.root=Path(model_dir).resolve()
        self.config=json.loads((self.root/"manifest.json").read_text())
        self.tokenizer=Tokenizer.from_file(str(self.root/"tokenizer.json"))
        options=ort.SessionOptions();options.intra_op_num_threads=threads
        options.inter_op_num_threads=1
        self.sessions={key:ort.InferenceSession(str(self.root/file),options,
            providers=["CPUExecutionProvider"]) for key,file in self.config["graphs"].items()}
        self.voice={}
        for key,spec in self.config["tensors"].items():
            dtype="<i8" if spec["dtype"]=="int64" else "<f4"
            self.voice[key]=np.fromfile(self.root/spec["file"],dtype=dtype).reshape(spec["shape"])
        self.last_trace=[]

    def token_ids(self,text):
        return [255]+self.tokenizer.encode(normalize(text).replace(" ","[SPACE]")).ids+[0]

    def _embeddings(self,ids,pos,kinds,exaggeration):
        return self.sessions["embeddings"].run(None,{
            "input_ids":ids,"position_ids":pos,"token_kind":kinds,
            "exaggeration":np.array(exaggeration,np.float32)})

    def _generate(self,text,rng,temperature,cfg_weight,repetition_penalty,max_tokens,exaggeration):
        tokens=np.array([self.token_ids(text)]*2,dtype=np.int64)
        text_length=tokens.shape[1]
        ids=np.concatenate([tokens,np.full((2,2),6561,np.int64)],axis=1)
        positions=np.concatenate([np.tile(np.arange(text_length),(2,1)),np.zeros((2,2),np.int64)],axis=1)
        kinds=np.concatenate([np.zeros_like(tokens),np.ones((2,2),np.int64)],axis=1)
        embedded,emotion=self._embeddings(ids,positions,kinds,exaggeration)
        prefix=np.concatenate([self.voice["conditioning"],emotion],axis=1)
        embedded=np.concatenate([np.repeat(prefix,2,axis=0),embedded],axis=1)
        prefix_length=prefix.shape[1]
        analyzer=AlignmentStreamAnalyzer((prefix_length,prefix_length+text_length),6562)
        cache={f"past_key_values.{layer}.{kind}":np.zeros((2,16,0,64),np.float32)
            for layer in range(30) for kind in ("key","value")}
        sequence=embedded.shape[1]
        generated=[6561]; eos=False
        for step in range(max_tokens):
            outputs=self.sessions["language_model"].run(None,{
                "inputs_embeds":embedded,"attention_mask":np.ones((2,sequence),np.int64),
                "cfg_weight":np.array(cfg_weight,np.float32),**cache})
            scores=analyzer.step(outputs[0][:,-1,:].copy(),outputs[1],next_token=generated[-1])[0]
            seen=np.unique(np.array(generated,np.int64))
            scores[seen]=np.where(scores[seen]<0,scores[seen]*repetition_penalty,scores[seen]/repetition_penalty)
            if temperature == 0:
                chosen=int(np.argmax(scores))
            else:
                scores=scores/temperature
                scores[scores < scores.max()+np.log(.05)]=-np.inf
                probabilities=np.exp(scores-scores.max()).astype(np.float64)
                probabilities/=probabilities.sum()
                chosen=int(rng.choice(len(scores),p=probabilities))
            generated.append(chosen)
            if chosen==6562:
                eos=True;break
            ids=np.full((2,1),chosen,np.int64)
            embedded,_=self._embeddings(ids,np.full_like(ids,step+1),np.ones_like(ids),exaggeration)
            sequence+=1
            cache=dict(zip(cache,outputs[2:]))
        if not eos:
            raise RuntimeError(f"Generation reached {max_tokens} tokens without EOS")
        result=np.array([[x for x in generated[1:] if x<6561]],dtype=np.int64)
        if result.size==0:raise RuntimeError("No speech tokens generated")
        self.last_trace.append({"text":text,"speech_tokens":result.shape[1],"eos":eos})
        return result

    def synthesize(self,text,seed=42,temperature=.8,cfg_weight=.5,repetition_penalty=2.0,
                   max_tokens=1000,exaggeration=.5):
        if not 0<=temperature<=2:raise ValueError("temperature must be 0..2")
        self.last_trace=[];audio=[]
        for index,chunk in enumerate(split_text(text)):
            rng=np.random.default_rng(seed+index)
            tokens=self._generate(chunk,rng,temperature,cfg_weight,repetition_penalty,max_tokens,exaggeration)
            combined=np.concatenate([self.voice["prompt_tokens"],tokens],axis=1)
            frames=combined.shape[1]*2
            samples=(frames-self.voice["speaker_features"].shape[1])*480
            wav=self.sessions["decoder"].run(None,{
                "speech_tokens":combined,"speaker_embeddings":self.voice["speaker_embeddings"],
                "speaker_features":self.voice["speaker_features"],
                "flow_noise":rng.standard_normal((1,80,frames)).astype(np.float32),
                "source_noise":rng.standard_normal((1,9,samples)).astype(np.float32)})[0].reshape(-1)
            if not np.isfinite(wav).all():raise RuntimeError("Decoder produced non-finite audio")
            if audio:audio.append(np.zeros(3600,np.float32))
            audio.append(wav)
        return np.concatenate(audio),24000


def save_wav(path,audio,sample_rate):
    pcm=(np.clip(audio,-1,1)*32767).astype("<i2")
    with wave.open(str(path),"wb") as file:
        file.setnchannels(1);file.setsampwidth(2);file.setframerate(sample_rate)
        file.writeframes(pcm.tobytes())


def main():
    parser=argparse.ArgumentParser(description="Standalone Uzbek TTS using ONNX Runtime")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text");group.add_argument("--text-file",type=Path)
    parser.add_argument("--model",type=Path,default=Path(__file__).resolve().parent.parent/"models")
    parser.add_argument("--output",type=Path,default=Path("speech.wav"))
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--temperature",type=float,default=.8)
    parser.add_argument("--max-tokens",type=int,default=1000)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args();start=time.time()
    tts=UzbekTTS(args.model,args.threads)
    audio,rate=tts.synthesize(args.text if args.text is not None else args.text_file.read_text(encoding="utf-8"),
        seed=args.seed,temperature=args.temperature,max_tokens=args.max_tokens)
    save_wav(args.output,audio,rate)
    print(json.dumps({"output":str(args.output),"audio_seconds":len(audio)/rate,
        "elapsed_seconds":round(time.time()-start,2),"chunks":tts.last_trace},ensure_ascii=False))

if __name__=="__main__":main()
