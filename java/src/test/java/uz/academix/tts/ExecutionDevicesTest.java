package uz.academix.tts;
import ai.onnxruntime.*;
import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
class ExecutionDevicesTest {
 @Test void parsesDevicesAndRejectsTypos() {
  assertEquals(ExecutionDevices.Device.MPS,ExecutionDevices.Device.parse("MPS"));
  assertEquals(ExecutionDevices.Device.AUTO,ExecutionDevices.Device.parse("auto"));
  assertThrows(IllegalArgumentException.class,()->ExecutionDevices.Device.parse("cude"));
 }
 @Test void cpuNeverChoosesGpu() {
  assertEquals(ExecutionDevices.Device.CPU,ExecutionDevices.select(ExecutionDevices.Device.CPU,EnumSet.allOf(OrtProvider.class),"Mac OS X"));
 }
 @Test void autoPrefersCudaThenAppleThenCpu() {
  assertEquals(ExecutionDevices.Device.CUDA,ExecutionDevices.select(ExecutionDevices.Device.AUTO,EnumSet.of(OrtProvider.CPU,OrtProvider.CUDA),"Linux"));
  assertEquals(ExecutionDevices.Device.MPS,ExecutionDevices.select(ExecutionDevices.Device.AUTO,EnumSet.of(OrtProvider.CPU,OrtProvider.WEBGPU),"Mac OS X"));
  assertEquals(ExecutionDevices.Device.CPU,ExecutionDevices.select(ExecutionDevices.Device.AUTO,EnumSet.of(OrtProvider.CPU),"Linux"));
 }
 @Test void explicitGpuNeverSilentlyFallsBack() {
  assertThrows(IllegalArgumentException.class,()->ExecutionDevices.select(ExecutionDevices.Device.CUDA,EnumSet.of(OrtProvider.CPU),"Linux"));
  assertThrows(IllegalArgumentException.class,()->ExecutionDevices.select(ExecutionDevices.Device.MPS,EnumSet.of(OrtProvider.CPU),"Mac OS X"));
  assertThrows(IllegalArgumentException.class,()->ExecutionDevices.select(ExecutionDevices.Device.MPS,EnumSet.of(OrtProvider.CPU,OrtProvider.CORE_ML),"Linux"));
 }
}
