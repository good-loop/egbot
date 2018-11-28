package com.goodloop.egbot.server;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

/**
 * Training a trivial linear model.
 */
public class TrainJavaTFExample {
  public static void main(String[] args) throws Exception {
	String graphPath = System.getProperty("user.dir") + "/data/models/testing/graph.pb";
    final byte[] graphDef = Files.readAllBytes(Paths.get(graphPath));
	final String checkpointDir = System.getProperty("user.dir") + "/data/models/testing/checkpoint";
    final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

    try (Graph graph = new Graph();
        Session sess = new Session(graph);
        Tensor<String> checkpointPrefix =
            Tensors.create(Paths.get(checkpointDir, "ckpt").toString())) {
      graph.importGraphDef(graphDef);

      // Initialize or restore.
      // The names of the tensors in the graph are printed out by the program
      // that created the graph:
      // https://github.com/tensorflow/models/blob/master/samples/languages/java/training/model/create_graph.py
      if (checkpointExists) {
        sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
      } else {
        sess.runner().addTarget("init").run();
      }
      System.out.print("Starting from       : ");
      printVariables(sess);

      // Train a bunch of times.
      // (Will be much more efficient if we sent batches instead of individual values).
      final Random r = new Random();
      final int NUM_EXAMPLES = 500;
      for (int i = 1; i <= 5; i++) {
        for (int n = 0; n < NUM_EXAMPLES; n++) {
          float in = r.nextFloat();
          try (Tensor<Float> input = Tensors.create(in);
              Tensor<Float> target = Tensors.create(3 * in + 2)) {
            // Again the tensor names are from the program that created the graph.
            // https://github.com/tensorflow/models/blob/master/samples/languages/java/training/model/create_graph.py
            sess.runner().feed("input", input).feed("target", target).addTarget("train").run();
          }
        }
        System.out.printf("After %5d examples: ", i*NUM_EXAMPLES);
        printVariables(sess);
      }

      // Checkpoint.
      // The feed and target name are from the program that created the graph.
      // https://github.com/tensorflow/models/blob/master/samples/languages/java/training/model/create_graph.py.
      sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();
      System.out.println(checkpointDir);
      
      // Example of "inference" in the same graph:
      try (Tensor<Float> input = Tensors.create(1.0f);
          Tensor<Float> output =
              sess.runner().feed("input", input).fetch("output").run().get(0).expect(Float.class)) {
        System.out.printf(
            "For input %f, produced %f (ideally would produce 3*%f + 2)\n",
            input.floatValue(), output.floatValue(), input.floatValue());
      }
    }
  }

  private static void printVariables(Session sess) {
    List<Tensor<?>> values = sess.runner().fetch("W/read").fetch("b/read").run();
    System.out.printf("W = %f\tb = %f\n", values.get(0).floatValue(), values.get(1).floatValue());
    for (Tensor<?> t : values) {
      t.close();
    }
  }
}
