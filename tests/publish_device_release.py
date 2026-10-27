"""Publish only after real CPU/Metal tests and preserve the previous release."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import struct
import subprocess
import time
import wave
import math
import zipfile

ROOT=Path(__file__).resolve().parent.parent

def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()

def clone(source,target):
    result=subprocess.run(["cp","-c",str(source),str(target)],capture_output=True)
    if result.returncode:shutil.copy2(source,target)

def main():
    reports={name:json.loads((ROOT/f"docs/device-{name}-result.json").read_text()) for name in ("cpu-device","auto-metal","auto-cpu-fallback","jar-mps","cuda")}
    for name in ("cpu-device","auto-metal","auto-cpu-fallback","jar-mps"):assert reports[name]["status"]=="passed",name
    assert reports["cuda"]["status"]=="unavailable_as_expected"
    assert reports["auto-metal"]["result"]["device"]=="mps"
    assert reports["auto-cpu-fallback"]["result"]["device"]=="cpu"
    assert reports["auto-cpu-fallback"]["result"].get("fallback_reason")
    assert all(reports["jar-mps"]["profile_provider_events"][graph].get("WebGpuExecutionProvider",0)>0 for graph in ("embeddings","language_model","decoder"))
    arrays=[]
    for name in ("cpu-device","mps-metal"):
        with wave.open(str(ROOT/f"samples/device-{name}.wav")) as audio:
            frames=audio.getnframes();arrays.append(struct.unpack("<"+"h"*frames,audio.readframes(frames)))
    a,b=arrays;assert len(a)==len(b)
    cosine=sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    assert cosine>.999
    built=ROOT/"java/target/uzbek-tts-0.1.0.jar"
    shutdown=json.loads((ROOT/"docs/validation-shutdown-mps.json").read_text())
    assert shutdown["status"]=="passed" and shutdown.get("jar_sha256")==sha(built)
    with zipfile.ZipFile(built) as jar:
        assert b"--device" in jar.read("uz/academix/tts/Main.class")
        assert b"RuntimeResources" in jar.read("ai/onnxruntime/OrtEnvironment$OrtEnvCloser.class")
        assert "ai/onnxruntime/native/linux-x64/libonnxruntime_providers_cuda.so" in jar.namelist()
        assert "ai/onnxruntime/native/win-x64/onnxruntime_providers_cuda.dll" in jar.namelist()
        assert "ai/onnxruntime/native/osx-aarch64/libonnxruntime.dylib" in jar.namelist()
        assert "models/language_model.onnx" in jar.namelist()
        gpu=Path.home()/".m2/repository/com/microsoft/onnxruntime/onnxruntime_gpu/1.23.2/onnxruntime_gpu-1.23.2.jar"
        with zipfile.ZipFile(gpu) as original:
            for name in ("ai/onnxruntime/native/linux-x64/libonnxruntime.so","ai/onnxruntime/native/win-x64/onnxruntime.dll"):
                assert jar.getinfo(name).CRC==original.getinfo(name).CRC
    release=ROOT/"dist/uzbek-tts-0.1.0.jar"
    backups=ROOT/"build/backups";backups.mkdir(parents=True,exist_ok=True)
    backup=backups/f"uzbek-tts-before-device-{time.time_ns()}.jar"
    clone(release,backup)
    temporary=release.with_name(".device-update-"+str(time.time_ns())+".jar")
    clone(built,temporary)
    digest=sha(built);assert sha(temporary)==digest
    os.replace(temporary,release)
    (ROOT/"dist/SHA256SUMS").write_text(digest+"  "+release.name+"\n")
    report={"status":"passed","jar":str(release),"jar_bytes":release.stat().st_size,"sha256":digest,
        "previous_release_backup":str(backup),"cpu_gpu_waveform_cosine":cosine,
        "devices":["cpu","auto","mps","cuda"],"default_device":"auto",
        "apple_backend":"Metal via native WebGPU","cuda_native_bundled_for":["linux-x64","win-x64"],
        "cuda_hardware_tested":False,"real_audio_modes_tested":["cpu","mps","auto"],
        "auto_cpu_fallback_tested":True,"gpu_http_clean_shutdown_tested":shutdown["runs"],"gpu_profile_events":reports["jar-mps"]["profile_provider_events"]}
    (ROOT/"docs/validation-devices.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
