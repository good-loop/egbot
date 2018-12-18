package egbot;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

import com.winterwell.bob.BuildTask;
import com.winterwell.bob.tasks.MavenDependencyTask;

public class GetTensorFlow extends BuildTask {

	public static void main(String[] args) throws Exception {

	    try (Graph g = new Graph()) {
	      final String value = "Hello from " + TensorFlow.version();

	      // Construct the computation graph with a single operation, a constant
	      // named "MyConst" with a value "value".
	      try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
	        // The Java API doesn't yet include convenience functions for adding operations.
	        g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
	      }

	      // Execute the "MyConst" operation in a Session.
	      try (Session s = new Session(g);
	          // Generally, there may be multiple output tensors,
	          // all of them must be closed to prevent resource leaks.
	          Tensor output = s.runner().fetch("MyConst").run().get(0)) {
	        System.out.println(new String(output.bytesValue(), "UTF-8"));
	      }
	    }

	}
	
	@Test
	@Override
	public void doTask() throws Exception {
		MavenDependencyTask mdt = new MavenDependencyTask();
		mdt.addDependency("org.tensorflow", "tensorflow", "1.11.0");
		mdt.addDependency("org.deeplearning4j","deeplearning4j-core","1.0.0-beta2");
		mdt.run();
	}

}
