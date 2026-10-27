package uz.academix.tts;

import ai.onnxruntime.*;
import com.fasterxml.jackson.databind.*;
import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/** Complete offline inference; requires only this JAR and Java 17+, never Python. */
public final class UzbekTts implements AutoCloseable {
    public record Options(long seed,float temperature,float cfgWeight,float repetitionPenalty,float exaggeration,int maxTokens) {
        public Options {
            if(seed<0||seed>4294967295L)throw new IllegalArgumentException("seed must be an unsigned 32-bit integer");
            if(!Float.isFinite(temperature)||temperature<0||temperature>2)throw new IllegalArgumentException("temperature must be 0..2");
            if(!Float.isFinite(cfgWeight)||cfgWeight<0||cfgWeight>1)throw new IllegalArgumentException("cfg_weight must be 0..1");
            if(!Float.isFinite(repetitionPenalty)||repetitionPenalty<2||repetitionPenalty>10)throw new IllegalArgumentException("repetition_penalty must be 2..10");
            if(!Float.isFinite(exaggeration)||exaggeration<0||exaggeration>2)throw new IllegalArgumentException("exaggeration must be 0..2");
            if(maxTokens<1||maxTokens>1000)throw new IllegalArgumentException("maxTokens must be 1..1000");
        }
        public static Options defaults(){return new Options(42,.8f,.5f,2f,.5f,1000);}
    }
    public record Audio(float[] samples,int sampleRate,List<int[]> tokens) {
        public byte[] wav() {
            int length=Math.multiplyExact(samples.length,2);
            ByteBuffer out=ByteBuffer.allocate(44+length).order(ByteOrder.LITTLE_ENDIAN);
            out.put(new byte[]{'R','I','F','F'}).putInt(36+length).put(new byte[]{'W','A','V','E'});
            out.put(new byte[]{'f','m','t',' '}).putInt(16).putShort((short)1).putShort((short)1);
            out.putInt(sampleRate).putInt(sampleRate*2).putShort((short)2).putShort((short)16);
            out.put(new byte[]{'d','a','t','a'}).putInt(length);
            for(float f:samples)out.putShort((short)(Math.max(-1,Math.min(1,f))*32767));
            return out.array();
        }
    }
    private record Embedded(float[] data,float[] emotion,int length){}
    private record TensorData(float[] floats,long[] longs,long[] shape){}
    private final OrtEnvironment environment=OrtEnvironment.getEnvironment("uzbek-tts");
    private final OrtSession embeddings,languageModel,decoder;
    private final ExecutionDevices.Device requestedDevice,effectiveDevice;
    private final String fallbackReason;
    private boolean closed;
    private record Sessions(OrtSession embeddings,OrtSession languageModel,OrtSession decoder){}
    private final TextTokenizer tokenizer;
    private final TensorData conditioning,prompt,speaker,features;
    private static final int DIM=1024,VOCAB=8194;

    public UzbekTts() throws Exception {this(null,null,4,"auto",null);}
    public UzbekTts(Path models,Path cache,int threads) throws Exception {this(models,cache,threads,"auto",null);}
    public UzbekTts(Path models,Path cache,int threads,String device) throws Exception {this(models,cache,threads,device,null);}
    public UzbekTts(Path models,Path cache,int threads,String device,Path profile) throws Exception {
        if(threads<1)throw new IllegalArgumentException("threads must be positive");
        requestedDevice=ExecutionDevices.Device.parse(device);
        ExecutionDevices.Device selected=ExecutionDevices.select(requestedDevice,OrtEnvironment.getAvailableProviders(),System.getProperty("os.name"));
        Path root=ModelBundle.resolve(models,cache);
        JsonNode config=new ObjectMapper().readTree(root.resolve("manifest.json").toFile());
        tokenizer=new TextTokenizer(root.resolve("tokenizer.json"));
        conditioning=read(root,config,"conditioning");prompt=read(root,config,"prompt_tokens");
        speaker=read(root,config,"speaker_embeddings");features=read(root,config,"speaker_features");
        Sessions sessions;String reason="";
        try {sessions=openSessions(root,config,cache,threads,selected,profile);}
        catch(Exception|LinkageError failure) {
            if(requestedDevice!=ExecutionDevices.Device.AUTO||selected==ExecutionDevices.Device.CPU)
                throw new IllegalStateException("Cannot initialize --device "+selected.flag+" ("+ExecutionDevices.provider(selected)+"). "+
                    (selected==ExecutionDevices.Device.CUDA?"A compatible NVIDIA driver, CUDA 12.8+ and cuDNN 9 runtime are required. ":"")+
                    "Cause: "+failure.getMessage()+". Use --device cpu for CPU inference.",failure);
            reason="Automatic "+selected.flag+" initialization failed: "+failure.getMessage();
            System.err.println(reason+"; falling back to CPU.");
            selected=ExecutionDevices.Device.CPU;
            sessions=openSessions(root,config,cache,threads,selected,profile);
        }
        effectiveDevice=selected;fallbackReason=reason;
        embeddings=sessions.embeddings();languageModel=sessions.languageModel();decoder=sessions.decoder();
        try{RuntimeResources.register(this);}catch(RuntimeException failure){close();throw failure;}
    }
    private Sessions openSessions(Path root,JsonNode config,Path cache,int threads,ExecutionDevices.Device device,Path profile)throws Exception {
        List<OrtSession> opened=new ArrayList<>();
        try {
            for(String graph:List.of("embeddings","language_model","decoder")) {
                try(OrtSession.SessionOptions options=new OrtSession.SessionOptions()) {
                    options.setIntraOpNumThreads(threads);options.setInterOpNumThreads(1);
                    options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                    ExecutionDevices.configure(options,device,root,cache,graph);
                    if(profile!=null){Files.createDirectories(profile);options.enableProfiling(profile.toAbsolutePath().resolve(graph).toString());}
                    opened.add(environment.createSession(root.resolve(config.path("graphs").path(graph).asText()).toString(),options));
                }
            }
            return new Sessions(opened.get(0),opened.get(1),opened.get(2));
        } catch(Exception|LinkageError failure) {
            for(OrtSession session:opened)try{session.close();}catch(Exception closeFailure){failure.addSuppressed(closeFailure);}
            throw failure;
        }
    }
    public Map<String,Object> deviceInfo() {
        Map<String,Object> info=new LinkedHashMap<>();
        info.put("requested_device",requestedDevice.flag);info.put("device",effectiveDevice.flag);
        info.put("provider",ExecutionDevices.provider(effectiveDevice));
        info.put("onnxruntime_version",environment.getVersion());
        if(effectiveDevice==ExecutionDevices.Device.MPS){
            if(ExecutionDevices.useWebGpu())info.put("backend","Metal");else info.put("compute_units","CPUAndGPU");
        }
        if(effectiveDevice!=ExecutionDevices.Device.CPU)info.put("cpu_operator_fallback",true);
        if(!fallbackReason.isEmpty())info.put("fallback_reason",fallbackReason);
        return Collections.unmodifiableMap(info);
    }
    private static TensorData read(Path root,JsonNode config,String name)throws IOException {
        JsonNode spec=config.path("tensors").path(name);
        long[] shape=new long[spec.path("shape").size()];for(int i=0;i<shape.length;i++)shape[i]=spec.path("shape").get(i).asLong();
        ByteBuffer bytes=ByteBuffer.wrap(Files.readAllBytes(root.resolve(spec.path("file").asText()))).order(ByteOrder.LITTLE_ENDIAN);
        if(spec.path("dtype").asText().equals("int64")){long[] data=new long[bytes.remaining()/8];bytes.asLongBuffer().get(data);return new TensorData(null,data,shape);}
        float[] data=new float[bytes.remaining()/4];bytes.asFloatBuffer().get(data);return new TensorData(data,null,shape);
    }
    private OnnxTensor floats(float[] data,long... shape)throws OrtException{return OnnxTensor.createTensor(environment,FloatBuffer.wrap(data),shape);}
    private OnnxTensor longs(long[] data,long... shape)throws OrtException{return OnnxTensor.createTensor(environment,LongBuffer.wrap(data),shape);}
    private static float[] array(OnnxValue value){FloatBuffer buffer=((OnnxTensor)value).getFloatBuffer();float[] out=new float[buffer.remaining()];buffer.get(out);return out;}
    private static float[] array(OrtSession.Result result,String name){return array(result.get(name).orElseThrow());}

    private Embedded embed(long[] ids,long[] positions,long[] kinds,int sequence,float exaggeration)throws OrtException {
        try(OnnxTensor a=longs(ids,2,sequence);OnnxTensor b=longs(positions,2,sequence);
            OnnxTensor c=longs(kinds,2,sequence);OnnxTensor d=floats(new float[]{exaggeration});
            OrtSession.Result result=embeddings.run(Map.of("input_ids",a,"position_ids",b,"token_kind",c,"exaggeration",d))) {
            return new Embedded(array(result,"inputs_embeds"),array(result,"emotion_embed"),sequence);
        }
    }
    private int[] generate(String text,Random random,Options options)throws OrtException {
        long[] textIds=tokenizer.encode(text);int count=textIds.length,sequence=count+2;
        long[] ids=new long[2*sequence],positions=new long[2*sequence],kinds=new long[2*sequence];
        for(int row=0;row<2;row++) {
            System.arraycopy(textIds,0,ids,row*sequence,count);
            for(int i=0;i<count;i++)positions[row*sequence+i]=i;
            for(int i=count;i<sequence;i++){ids[row*sequence+i]=6561;kinds[row*sequence+i]=1;}
        }
        Embedded input=embed(ids,positions,kinds,sequence,options.exaggeration());
        int prefix=(int)conditioning.shape()[1]+1;
        float[] embedded=new float[2*(prefix+sequence)*DIM];
        for(int row=0;row<2;row++) {
            int offset=row*(prefix+sequence)*DIM;
            System.arraycopy(conditioning.floats(),0,embedded,offset,conditioning.floats().length);
            System.arraycopy(input.emotion(),0,embedded,offset+conditioning.floats().length,DIM);
            System.arraycopy(input.data(),row*sequence*DIM,embedded,offset+prefix*DIM,sequence*DIM);
        }
        sequence+=prefix;int query=sequence;
        Alignment alignment=new Alignment(prefix,prefix+count,6562);
        List<Integer> generated=new ArrayList<>();generated.add(6561);
        OrtSession.Result prior=null;boolean eos=false;
        try(OnnxTensor empty=floats(new float[0],2,16,0,64)) {
            Map<String,OnnxTensor> cache=new LinkedHashMap<>();
            for(int i=0;i<30;i++)for(String kind:List.of("key","value"))cache.put("past_key_values."+i+"."+kind,empty);
            try {
                for(int step=0;step<options.maxTokens();step++) {
                    long[] mask=new long[2*sequence];Arrays.fill(mask,1);
                    OrtSession.Result result;
                    try(OnnxTensor a=floats(embedded,2,query,DIM);OnnxTensor b=longs(mask,2,sequence);
                        OnnxTensor c=floats(new float[]{options.cfgWeight()})) {
                        Map<String,OnnxTensor> feed=new HashMap<>(cache);
                        feed.put("inputs_embeds",a);feed.put("attention_mask",b);feed.put("cfg_weight",c);
                        result=languageModel.run(feed);
                    }
                    if(prior!=null)prior.close();prior=result;
                    float[] all=array(prior,"logits"),scores=Arrays.copyOfRange(all,all.length-VOCAB,all.length);
                    alignment.apply(scores,array(prior,"attn_layers"),query,sequence,generated.get(generated.size()-1));
                    Set<Integer> seen=new HashSet<>(generated);
                    for(int token:seen)scores[token]=scores[token]<0?scores[token]*options.repetitionPenalty():scores[token]/options.repetitionPenalty();
                    int next=sample(scores,random,options.temperature());generated.add(next);
                    if(next==6562){eos=true;break;}
                    Embedded nextInput=embed(new long[]{next,next},new long[]{step+1,step+1},new long[]{1,1},1,options.exaggeration());
                    embedded=nextInput.data();query=1;sequence++;
                    for(int layer=0;layer<30;layer++)for(String kind:List.of("key","value"))
                        cache.put("past_key_values."+layer+"."+kind,(OnnxTensor)prior.get("present."+layer+"."+kind).orElseThrow());
                }
            } finally {if(prior!=null)prior.close();}
        }
        if(!eos)throw new IllegalStateException("Generation reached the token limit without EOS");
        return generated.stream().skip(1).filter(v->v<6561).mapToInt(Integer::intValue).toArray();
    }
    private static int sample(float[] scores,Random random,float temperature) {
        int best=0;for(int i=1;i<scores.length;i++)if(scores[i]>scores[best])best=i;
        if(temperature==0)return best;
        double max=scores[best]/temperature,threshold=max+Math.log(.05),sum=0;
        double[] probabilities=new double[scores.length];
        for(int i=0;i<scores.length;i++) {
            double score=scores[i]/temperature;
            if(score>=threshold){probabilities[i]=Math.exp(score-max);sum+=probabilities[i];}
        }
        double draw=random.nextDouble()*sum;
        for(int i=0;i<probabilities.length;i++){draw-=probabilities[i];if(draw<=0)return i;}
        return best;
    }
    private static float[] noise(Random random,int count){float[] data=new float[count];for(int i=0;i<count;i++)data[i]=(float)random.nextGaussian();return data;}
    private float[] decode(int[] generated,Random random)throws OrtException {
        if(generated.length==0)throw new IllegalStateException("No speech tokens generated");
        long[] tokens=Arrays.copyOf(prompt.longs(),prompt.longs().length+generated.length);
        for(int i=0;i<generated.length;i++)tokens[prompt.longs().length+i]=generated[i];
        int frames=tokens.length*2,samples=(frames-(int)features.shape()[1])*480;
        try(OnnxTensor a=longs(tokens,1,tokens.length);OnnxTensor b=floats(speaker.floats(),speaker.shape());
            OnnxTensor c=floats(features.floats(),features.shape());
            OnnxTensor d=floats(noise(random,80*frames),1,80,frames);OnnxTensor e=floats(noise(random,9*samples),1,9,samples);
            OrtSession.Result result=decoder.run(Map.of("speech_tokens",a,"speaker_embeddings",b,"speaker_features",c,"flow_noise",d,"source_noise",e))) {
            float[] audio=array(result,"waveform");
            for(float f:audio)if(!Float.isFinite(f))throw new IllegalStateException("Non-finite audio generated");
            return audio;
        }
    }
    public synchronized Audio synthesize(String text,Options options)throws OrtException {
        if(closed)throw new IllegalStateException("TTS model is closed");
        List<float[]> pieces=new ArrayList<>();List<int[]> tokenChunks=new ArrayList<>();int size=0,index=0;
        for(String chunk:TextTokenizer.split(text)) {
            Random random=new Random(options.seed()+index++);
            int[] tokens=generate(chunk,random,options);tokenChunks.add(tokens);
            float[] audio=decode(tokens,random);
            if(!pieces.isEmpty()){pieces.add(new float[3600]);size+=3600;}
            pieces.add(audio);size+=audio.length;
        }
        float[] audio=new float[size];int offset=0;
        for(float[] piece:pieces){System.arraycopy(piece,0,audio,offset,piece.length);offset+=piece.length;}
        return new Audio(audio,24000,List.copyOf(tokenChunks));
    }
    public byte[] synthesize(String text)throws OrtException{return synthesize(text,Options.defaults()).wav();}
    @Override public synchronized void close()throws OrtException {
        if(closed)return;
        closed=true;
        OrtException failure=null;
        for(OrtSession session:List.of(decoder,languageModel,embeddings))try{session.close();}
        catch(OrtException problem){if(failure==null)failure=problem;else failure.addSuppressed(problem);}
        RuntimeResources.unregister(this);
        if(failure!=null)throw failure;
    }
}
