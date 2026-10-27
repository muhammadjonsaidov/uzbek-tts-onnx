package uz.academix.tts;

import ai.onnxruntime.*;
import java.io.IOException;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;

/** Device selection is explicit; provider registration is not a hardware guarantee. */
final class ExecutionDevices {
    enum Device {
        AUTO("auto"), CPU("cpu"), MPS("mps"), CUDA("cuda");
        final String flag;
        Device(String flag){this.flag=flag;}
        static Device parse(String value) {
            if(value!=null)for(Device d:values())if(d.flag.equals(value.strip().toLowerCase(Locale.ROOT)))return d;
            throw new IllegalArgumentException("Invalid --device '"+value+"'; use auto, cpu, mps, or cuda");
        }
    }
    static boolean useWebGpu(){return System.getProperty("uzbek.tts.apple.provider","webgpu").equals("webgpu");}
    static boolean appleAvailable(Set<OrtProvider> available){return available.contains(useWebGpu()?OrtProvider.WEBGPU:OrtProvider.CORE_ML);}
    static Device select(Device requested,Set<OrtProvider> available,String osName) {
        boolean mac=osName.toLowerCase(Locale.ROOT).contains("mac")||osName.toLowerCase(Locale.ROOT).contains("darwin");
        if(requested==Device.AUTO) {
            if(available.contains(OrtProvider.CUDA))return Device.CUDA;
            if(mac&&appleAvailable(available))return Device.MPS;
            return Device.CPU;
        }
        if(requested==Device.MPS&&(!mac||!appleAvailable(available)))
            throw new IllegalArgumentException("--device mps requires macOS and a Apple GPU-enabled runtime; available providers: "+available);
        if(requested==Device.CUDA&&!available.contains(OrtProvider.CUDA))
            throw new IllegalArgumentException("--device cuda requires the CUDA provider on NVIDIA Linux/Windows; available providers: "+available);
        return requested;
    }
    static String provider(Device device) {
        return switch(device){case CPU->"CPUExecutionProvider";case MPS->useWebGpu()?"WebGpuExecutionProvider":"CoreMLExecutionProvider";case CUDA->"CUDAExecutionProvider";default->throw new IllegalArgumentException("auto must be resolved first");};
    }
    static void configure(OrtSession.SessionOptions options,Device device,Path models,Path cache,String graph)throws Exception {
        if(device==Device.CUDA)options.addCUDA(0);
        if(device==Device.MPS && useWebGpu()){options.addWebGPU(Map.of());return;}
        if(device==Device.MPS) {
            Path base=cache!=null?cache:Path.of(System.getProperty("user.home"),".cache","uzbek-tts");
            Path index=models.resolve("bundle-index.json");
            byte[] bytes=Files.exists(index)?Files.readAllBytes(index):models.toAbsolutePath().toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
            String hash=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)).substring(0,16);
            String format=System.getProperty("uzbek.tts.coreml.format","MLProgram");
            Path compiled=base.toAbsolutePath().resolve("coreml-"+OrtEnvironment.getEnvironment().getVersion()+"-"+format).resolve(hash).resolve(graph);
            Files.createDirectories(compiled);
            options.addCoreML(Map.of("ModelFormat",format,"MLComputeUnits","CPUAndGPU",
                "RequireStaticInputShapes","0","ModelCacheDirectory",compiled.toString()));
        }
    }
    static Map<String,Object> inventory() {
        Set<OrtProvider> providers=OrtEnvironment.getAvailableProviders();
        List<String> devices=new ArrayList<>();devices.add("cpu");
        if(System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("mac")&&appleAvailable(providers))devices.add("mps");
        if(providers.contains(OrtProvider.CUDA))devices.add("cuda");
        return Map.of("onnxruntime",OrtEnvironment.getEnvironment().getVersion(),
            "registered_providers",providers.stream().map(OrtProvider::getName).toList(),
            "device_options",devices,"note","GPU hardware and driver availability are checked when creating model sessions");
    }
}
