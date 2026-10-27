"""Copy exactly one JAR into an empty directory and generate real audio offline."""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
import wave
import zipfile

ROOT=Path(__file__).resolve().parent.parent

def copy_file(source,destination):
    result=subprocess.run(["cp","-c",str(source),str(destination)],capture_output=True)
    if result.returncode:shutil.copy2(source,destination)

def main():
    release=ROOT/"dist/uzbek-tts-0.1.0.jar"
    # Verification must never rewrite an artifact that another Java process may be using.
    if not release.is_file():raise FileNotFoundError("Publish dist/uzbek-tts-0.1.0.jar first")
    with zipfile.ZipFile(release,"r") as archive:
        entries=set(archive.namelist())
        for name in ("LICENSE","LICENSE.models","docs/onnx-recipe-README.md",
                     "uz/academix/tts/Main.class","ai/onnxruntime/OrtEnvironment.class",
                     "models/decoder.onnx"):
            assert name in entries,name
    isolated=Path(tempfile.mkdtemp(prefix="uzbek-tts-one-jar-"))
    copy_file(release,isolated/"uzbek-tts.jar")
    initial=sorted(path.name for path in isolated.iterdir())
    assert initial==["uzbek-tts.jar"]
    environment={k:v for k,v in os.environ.items() if k not in {
        "CLASSPATH","PYTHONPATH","VIRTUAL_ENV","JAVA_TOOL_OPTIONS","JDK_JAVA_OPTIONS"}}
    environment["PATH"]="/usr/bin:/bin"
    command=["/usr/bin/java","-jar","uzbek-tts.jar","--cache","cache",
        "--text","Salom, men o‘zbekcha gapiryapman.","--temperature","0",
        "--output","speech.wav","--trace","tokens.json"]
    started=time.time()
    result=subprocess.run(command,cwd=isolated,env=environment,capture_output=True,text=True,timeout=300)
    if result.returncode:
        raise RuntimeError(result.stdout+"\n"+result.stderr)
    print(result.stdout,flush=True)
    with wave.open(str(isolated/"speech.wav")) as audio:
        rate=audio.getframerate();channels=audio.getnchannels();width=audio.getsampwidth()
        frames=audio.getnframes();raw=audio.readframes(frames)
    values=struct.unpack("<"+"h"*(len(raw)//2),raw)
    rms=math.sqrt(sum(value*value for value in values)/len(values))
    assert rate==24000 and channels==1 and width==2 and frames>0 and rms>10
    shutil.copy2(isolated/"speech.wav",ROOT/"samples/standalone-jar.wav")
    report={"status":"passed","initial_files":initial,"isolated_directory":str(isolated),
        "command":command,"model_argument_supplied":False,"python_process_required":False,
        "old_projects_on_classpath":False,"sample_rate":rate,"audio_seconds":frames/rate,
        "rms_pcm16":rms,"elapsed_seconds":round(time.time()-started,2),
        "jar_bytes":release.stat().st_size,"java_result":json.loads(result.stdout.strip())}
    (ROOT/"docs/validation-standalone.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    digest=hashlib.sha256()
    with release.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    (ROOT/"dist/SHA256SUMS").write_text(digest.hexdigest()+"  "+release.name+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
