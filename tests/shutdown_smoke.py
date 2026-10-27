"""Real HTTP inference followed by SIGTERM and SIGINT; assert no native crash."""
from pathlib import Path
import argparse, hashlib, io, json, os, signal, subprocess, time, urllib.request, wave
ROOT=Path(__file__).resolve().parent.parent

def main():
    p=argparse.ArgumentParser();p.add_argument("--jar",type=Path);p.add_argument("--device",default="mps");p.add_argument("--full-api",action="store_true");a=p.parse_args()
    if a.jar:command=["/usr/bin/java","-jar",str(a.jar.resolve())]
    else:
        repo=Path.home()/".m2/repository"
        jars=[repo/"com/microsoft/onnxruntime/onnxruntime/1.23.2/onnxruntime-1.23.2.jar"]
        jars += [repo/f"com/fasterxml/jackson/core/{n}/2.18.2/{n}-2.18.2.jar" for n in ("jackson-databind","jackson-core","jackson-annotations")]
        command=["/usr/bin/java","-cp",os.pathsep.join(map(str,[ROOT/"java/target/classes",*jars])),"uz.academix.tts.Main"]
    env=dict(os.environ,TTS_API_KEY="standalone-smoke-key",TTS_CORS_ORIGINS="http://localhost:3000",TTS_TEST_EXPECT_DEVICE=a.device)
    reports=[]
    for sig in (signal.SIGTERM,signal.SIGINT):
        label=sig.name.lower();log=ROOT/f"docs/shutdown-{a.device}-{label}.log"
        with log.open("w") as stream:
            process=subprocess.Popen(command+["--device",a.device,"--model",str(ROOT/"models"),"--cache",str(ROOT/"build/device-cache"),"--serve","--port","18001"],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
            try:
                deadline=time.monotonic()+180
                while True:
                    if process.poll() is not None:raise AssertionError(log.read_text()[-4000:])
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:18001/ready",timeout=2) as response:ready=json.load(response)
                        assert ready["device"]==a.device;break
                    except (OSError,TimeoutError):
                        if time.monotonic()>deadline:raise TimeoutError("Server startup timeout")
                        time.sleep(.5)
                if a.full_api and sig==signal.SIGTERM:
                    subprocess.run(["/opt/homebrew/bin/python3",str(ROOT/"tests/http_smoke.py")],env=env,check=True,timeout=180)
                else:
                    request=urllib.request.Request("http://127.0.0.1:18001/v1/audio/speech",data=json.dumps({"model":"uzbek-chatterbox","input":"Ha.","temperature":0}).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer standalone-smoke-key"})
                    with urllib.request.urlopen(request,timeout=120) as response:data=response.read()
                    with wave.open(io.BytesIO(data)) as audio:assert audio.getframerate()==24000 and audio.getnframes()>0
                started=time.monotonic();process.send_signal(sig);code=process.wait(timeout=90)
                text=log.read_text()
                assert code in (0,128+int(sig),-int(sig)),(code,text[-6000:])
                assert not any(s in text for s in ("SIGSEGV","fatal error","Error closing","Exception")),text[-6000:]
                reports.append({"signal":sig.name,"returncode":code,"shutdown_seconds":round(time.monotonic()-started,2),"real_audio":True})
            finally:
                if process.poll() is None:process.kill();process.wait()
    report={"status":"passed","device":a.device,"runs":reports,"full_http_api":a.full_api}
    if a.jar:
        h=hashlib.sha256()
        with a.jar.open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
        report["jar_sha256"]=h.hexdigest()
    (ROOT/f"docs/validation-shutdown-{a.device}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
