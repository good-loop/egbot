package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.Hdf5Archive;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ResourceUtils;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.Tensors;

public class SampleTrainedLSTM {
	
	public static void main(String[] args) throws Exception {
		
		String seed = "what is a derivative";
		Object generatedAnswer = generateAnswerJavaTF(seed);
		System.out.println(generatedAnswer);
	}
	
	/*
	 * loading tensorflow graph (requires appropriate operation name)
	 */
	private static Object generateAnswerJavaTF(String q) throws IllegalArgumentException, IOException {
        // TODO how do we generate this file??
		String modelPath = System.getProperty("user.dir") + "/latest.pb";	
		try (Graph graph = new Graph()){
			graph.importGraphDef(Files.readAllBytes(Paths.get(modelPath)));
			try (Session sess = new Session(graph);
				Tensor<String> input = Tensors.create(q);
				Tensor<String> output = sess.runner()
						// label for input (this will be defined in the code that creates the graph)
						// TODO put in appropriate labels once the graph is trained
						.feed("x", input)
						// label for output
						.fetch("y")
						.run()
						.get(0).expect(String.class)){
			};
			graph.close();
		};
		// TODO: transform output from Tensor into String and return
		return "";
	}	
	
	/*
	 * abandoned attempt to load Keras model using DL4J but getting error "#000: H5D.c line 294 in H5Dopen2(): unable to open dataset"
	 */
	private static String loadModelDL4J() throws Exception {
		String modelPath = System.getProperty("user.dir") 
				+ "/data/models/final/v2/gen_sentences_lstm_model.final.hdf5";
		Hdf5Archive archive = new Hdf5Archive(modelPath);
		INDArray out = archive.readDataSet(modelPath);
		return "";

	}
	
	/*
	 * abandoned attempt to load Keras model using ND4J but getting error "Nd4jBackend$NoAvailableBackendException: Please ensure that you have an nd4j backend on your classpath" 
	 */
	private static String loadModelND4J() throws Exception {
		String modelPath = System.getProperty("user.dir") 
				+ "/data/models/final/v2/gen_sentences_lstm_model.final.hdf5";
		String simpleMlp = ResourceUtils.getFile(modelPath).getPath();
		MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(simpleMlp);
		INDArray input = Nd4j.create(256, 100);
		INDArray output = model.output(input);
		model.fit(input, output);
		System.out.println(output);
		return "";
	}

	/*
	 * test if TensorFlow works 
	 */
	private static void testJavaTF() throws Exception {
		try (Graph g = new Graph()) {
			final String value = "Hello from " + TensorFlow.version();
	    	try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
	    		g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
	    	}
	    	try (Session s = new Session(g); Tensor output = s.runner().fetch("MyConst").run().get(0)) {
	    		System.out.println(new String(output.bytesValue(), "UTF-8"));
	    	}
		}
	}
}
