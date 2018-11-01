package com.goodloop.egbot.server;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.types.UInt8;

/**
 * @testedby {@link TrainLSTMTest}
 * @author daniel
 *
 */
public class TrainLSTM {
	String[] vocab;
	
	TrainLSTM(){
		initVocab();
	}
	
	/*
	 * initialise vocabulary
	 */
	void initVocab(){	
		// TODO: write up proper vocab loading or building
		vocab = new String[] {"here", "is", "a", "question", "and", "there", "is", "an", "answer", "what", "probability", "the", "measure", "of", "likelihood", "that", "event", "will", "occur"};
	}
	
	/*
	 * train the model
	 */
	void train() throws IOException {	
		// graph obtained from running data-collection/build_graph/createLSTMGraphTF.py	
		final String graphPath = System.getProperty("user.dir") + "/data/models/final/v3/lstmGraphTF.pb";
		final byte[] graphDef = Files.readAllBytes(Paths.get(graphPath));
		final String checkpointDir = "/data/models/final/v3/checkpoint";
	    final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

	    try (Graph graph = new Graph();
	        Session sess = new Session(graph);
	        Tensor<String> checkpointPrefix =
	        Tensors.create(Paths.get(checkpointDir, "ckpt").toString())) {
	    	
	    	graph.importGraphDef(graphDef);
	    	// initialise or restore.
			// The names of the tensors and operations in the graph are printed out by the program that created the graph
	    	// you can find the names in the following file: data/models/final/v3/tensorNames.txt
			if (checkpointExists) {
				sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
			} else {
				sess.runner().addTarget("init").run();
			}
			System.out.print("Starting from       : ");
			//printVariables(sess);
		
			// train a bunch of times.
			// (will be much more efficient if we sent batches instead of individual values).
			String[] testInstances = new String[] {"here is a question", "what is probability"};
			String[] testTargets = new String[] {"and there is an answer", "probability is the measure of the likelihood that an event will occur"};
			int num_epochs = 5; // training epochs
			for (int i = 1; i <= num_epochs; i++) {
				for (int j = 0; j < testInstances.length; j++) {
					String testInstance = testInstances[j];
					String testTarget = testTargets[j];
					try (Tensor<?> instanceTensor = Tensors.create(wordsIntoFeatureVector(testInstance));
							Tensor<?> targetTensor = Tensors.create(wordsIntoFeatureVector(testTarget))) {
						// The names of the tensors and operations in the graph are printed out by the program that created the graph
				    	// you can find the names in the following file: data/models/final/v3/tensorNames.txt
						sess.runner().feed("input", instanceTensor).feed("target", targetTensor).addTarget("Adam").run(); // Adam is the name of the train operation because it uses Adam optimiser
						}
				}
				System.out.printf("After %d examples: ", i*testInstances.length);
				printVariables(sess);
			}

			// checkpoint
			sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();

			// inference 
			try (Tensor<?> input = Tensors.create(wordsIntoFeatureVector(testInstances[0]));
					Tensor<?> output =
							sess.runner().feed("input", input).fetch("output").run().get(0).expect(Float.class)) {
				System.out.printf(
						"Instance: %f \n Prediction: %f \n",
						input.toString(), output.toString());
			}     
	    }
	}
	
	/*
	 * sample from the model
	 */
	void sample() throws Exception {
		 String modelPath = System.getProperty("user.dir") + "/data/models/final/v3/lstmGraphTF.pb";
		 try (SavedModelBundle model = SavedModelBundle.load(modelPath, "serve")) {
			 printSignature(model);
			 String question = "what is probability";
			 List<Tensor<?>> outputs = null;
			 try (Tensor<Float> questionTensor = Tensors.create(wordsIntoFeatureVector(question))) {
				 outputs = model
				          .session()
				          .runner()
				          .feed("input", questionTensor)
				          .fetch("output")
				          .run();
			}
			try (Tensor<Float> answerTensor = outputs.get(0).expect(Float.class);) {
				String answer = featureVectorIntoWords(answerTensor);
			}			 
		 }
	}
	
	float[] wordsIntoFeatureVector(String words) {
		// TODO there should be a more efficient way of doing this
		
		String[] splitted = words.split("\\s+");
		float[] wordsOneHotEncoded = new float[1000]; // let's say the vocab size is 100000 (this needs to be defined when building the graph in python)
		Arrays.fill(wordsOneHotEncoded, 0);
		
		for (int i = 0; i < splitted.length; i++) {
			for (int j = 0; j < vocab.length; j++) {	
				if (vocab[j] == splitted[i]) {
					wordsOneHotEncoded[j] = (float) 1;
				} 
			}
		}
		
		return wordsOneHotEncoded;
	}
	
	String featureVectorIntoWords(Tensor<?> answerTensor) {
		// TODO transform tensor into words
		return null;
	}

	/*
	 * Prints out the signature of the model, which specifies what type of model is being exported, 
	 * and the input/output tensors to bind to when running inference.
	 */
	private static void printSignature(SavedModelBundle model) throws Exception {
		MetaGraphDef m = MetaGraphDef.parseFrom(model.metaGraphDef());
		SignatureDef sig = m.getSignatureDefOrThrow("serving_default");
		int numInputs = sig.getInputsCount();
		int i = 1;
		System.out.println("MODEL SIGNATURE");
		System.out.println("Inputs:");
		for (Map.Entry<String, TensorInfo> entry : sig.getInputsMap().entrySet()) {
		  TensorInfo t = entry.getValue();
		  System.out.printf(
		      "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
		      i++, numInputs, entry.getKey(), t.getName(), t.getDtype());
		}
		int numOutputs = sig.getOutputsCount();
		i = 1;
		System.out.println("Outputs:");
		for (Map.Entry<String, TensorInfo> entry : sig.getOutputsMap().entrySet()) {
		  TensorInfo t = entry.getValue();
		  System.out.printf(
		      "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
		      i++, numOutputs, entry.getKey(), t.getName(), t.getDtype());
		}
		System.out.println("-----------------------------------------------");
	}
	
	private static void printVariables(Session sess) {
	    List<Tensor<?>> values = sess.runner().fetch("W/read").fetch("b/read").run();
	    System.out.printf("W = %f\tb = %f\n", values.get(0).floatValue(), values.get(1).floatValue());
	    for (Tensor<?> t : values) {
	      t.close();
	    }
	}
}
