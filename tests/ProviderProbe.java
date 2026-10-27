import ai.onnxruntime.*;
import java.util.*;
class ProviderProbe {
 public static void main(String[] args)throws Exception {
  var environment=OrtEnvironment.getEnvironment();
  System.out.println("ONNX Runtime="+environment.getVersion());
  System.out.println("Available providers="+OrtEnvironment.getAvailableProviders());
  if(OrtEnvironment.getAvailableProviders().contains(OrtProvider.CORE_ML)) {
   try(var options=new OrtSession.SessionOptions()) {
    options.addCoreML(Map.of("ModelFormat","MLProgram","MLComputeUnits","CPUAndGPU","RequireStaticInputShapes","0"));
    System.out.println("CoreML CPUAndGPU registration=OK");
   }
  }
 }
}
