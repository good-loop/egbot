package com.goodloop.egbot.server;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.GPUOptions;

public class TestGPU {
  public static void main(String[] args) throws Exception {
	
	// setting session config to use the GPU
      GPUOptions gpuOptions = GPUOptions.newBuilder()
      		.setPerProcessGpuMemoryFraction(1)
              .setForceGpuCompatible(true)
              .setAllowGrowth(true)
              .build();
      
      ConfigProto config = ConfigProto.newBuilder()
      		.setLogDevicePlacement(true)
              .setGpuOptions(gpuOptions)
              .build();

  	//ConfigProto.Builder configBuilder = ConfigProto.parseFrom(config.toByteString()).toBuilder();
  	//configBuilder = configBuilder.putDeviceCount("GPU", 1);
	  
    try (Graph g = new Graph()) {
      final String value = "Hello from " + TensorFlow.version();

      // Construct the computation graph with a single operation, a constant
      // named "MyConst" with a value "value".
      try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
        // The Java API doesn't yet include convenience functions for adding operations.
        g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
      }

      // Execute the "MyConst" operation in a Session.
      try (Session s = new Session(g, config.toByteArray());
    		  
    		  
          // Generally, there may be multiple output tensors,
          // all of them must be closed to prevent resource leaks.
          Tensor output = s.runner().fetch("MyConst").run().get(0)) {
        System.out.println(new String(output.bytesValue(), "UTF-8"));
      }
    }
  }
}
