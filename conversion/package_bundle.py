"""Refresh the inference-only resource index before building the JAR."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parent.parent

def main():
    files=[]
    for path in sorted((ROOT/"models").iterdir()):
        if not path.is_file() or path.name=="bundle-index.json":continue
        digest=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
        files.append({"name":path.name,"size":path.stat().st_size,"sha256":digest.hexdigest()})
    for name in ("embeddings.onnx","language_model.onnx","decoder.onnx","manifest.json","tokenizer.json"):
        if name not in {f["name"] for f in files}:raise FileNotFoundError(name)
    (ROOT/"models/bundle-index.json").write_text(json.dumps({"format_version":1,"files":files},indent=2)+"\n")
    print("Indexed",len(files),"inference assets")

if __name__=="__main__":main()
