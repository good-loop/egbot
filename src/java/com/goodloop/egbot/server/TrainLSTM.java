package com.goodloop.egbot.server;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.types.UInt8;

import com.sun.xml.internal.bind.v2.runtime.unmarshaller.XsiNilLoader.Array;
import com.winterwell.utils.containers.Containers;

/**
 * @testedby {@link TrainLSTMTest}
 * @author daniel
 *
 */
public class TrainLSTM {
	String[] vocab;
	int vocab_size;
	int seq_length;
	int ckptVersion = 25;//new Random().nextInt(10000);
	
	TrainLSTM(){
		initVocab();
	}
	
	/**
	 * initialise vocabulary
	 */
	void initVocab(){	
		// TODO: write up proper vocab loading or building (vocab_size has to be the same in the script that constructs the graph)
		vocab = new String[] {"UNKNOWN", "and", "there", "is", "an", "answer", "here", "is", "a", "question", "what", "probability", "the", "measure", "of", "likelihood", "that", "event", "will", "occur"};
		vocab_size = vocab.length; 
		seq_length = 30;
	}
	
	/**
	 * train the model
	 */
	void train() throws IOException {	
		// graph obtained from running data-collection/build_graph/createLSTMGraphTF.py	
		final String graphPath = System.getProperty("user.dir") + "/data/models/final/v3/lstmGraphTF.pb";
		Path gp = Paths.get(graphPath);
		assert Files.exists(gp) : "No "+gp+" better run data-collection/build_graph/createLSTMGraphTF.py";
		final byte[] graphDef = Files.readAllBytes(gp);
		final String checkpointDir = System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + ckptVersion;
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
				System.out.println("Restoring model ...");
				sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
			} else {
				System.out.println("Initialising model ...");
				sess.runner().addTarget("init").run();
			}
			System.out.print("Starting from: \n");
			//printVariables(sess);

			//TODO: does the input have to be set (seq_length = 30)?
			String trainingData = "here is a question and there is an answer here is a question and there is an answer here is a question and there is an answer here is a question and there is an answer here is a question and there is an answer here is a question and there is an answer here is a question and there is an answer here is a question and there is an answer ";
			String[] trainingDataArray = tokenise(trainingData);
			String[] instanceArray = new String[seq_length];
			//System.arraycopy(trainingDataArray, 0, instanceArray, 0, seq_length);
			String target = ""; //trainingDataArray[seq_length];
			int noOfInstances = trainingDataArray.length-seq_length;
			
			// train a bunch of times.
			// (will be much more efficient if we sent batches instead of individual values).
			int num_epochs = 5; // training epochs
			for (int i = 1; i <= num_epochs; i++) {
				for (int j = 0; j < noOfInstances; j++) {
					System.arraycopy( trainingDataArray, j, instanceArray, 0, seq_length);
					target = trainingDataArray[seq_length+j];
					System.out.printf("Instance: %s\n Target: %s \n", Arrays.deepToString(instanceArray), target);
					
					// NB: try-with to ensure C level resources are released
					try (Tensor<?> instanceTensor = Tensors.create(wordsIntoInputVector(instanceArray));
							Tensor<?> targetTensor = Tensors.create(wordsIntoFeatureVector(target))) {
						// The names of the tensors and operations in the graph are printed out by the program that created the graph
				    	// you can find the names in the following file: data/models/final/v3/tensorNames.txt
						Runner runner = sess.runner().feed("input", instanceTensor).feed("target", targetTensor).addTarget("Adam");
//						runner.run(); // Adam is the name of the train operation because it uses Adam optimiser
						runner.runAndFetchMetadata();
					}
				}
				System.out.printf("After %d examples: \n", i*noOfInstances);
				//printVariables(sess);
			}

			// checkpoint
			sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();
			
			// inference 
			String testInstance = "here is a question and there is an answer here is a question and there is an answer here is a question and there is an answer here is a";
			try (Tensor<?> input = Tensors.create(wordsIntoInputVector(tokenise(testInstance)));
					Tensor<?> output =
							sess.runner().feed("input", input).fetch("output").run().get(0).expect(Float.class)) {
				float[][] outputArray = new float[1][vocab_size];
				output.copyTo(outputArray);
				String nextWord = mostLikelyWord(outputArray);
				System.out.printf(
						"Instance: %s \n Prediction: %s \n",
						testInstance, nextWord);
			}     
	    }
	}
	
	/**
	 * 
	 * @param testInstance
	 * @return e.g. [[7], [13], [12] ...] 30 long  to match seq_length=30
	 */
	private float[][] wordsIntoInputVector(String[] words) {
		//System.out.println("\n");
		//System.out.println(testInstance);		
		float[][] input = new float[30][1];
		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			int wordIndex = Containers.indexOf(word, vocab);
			if (wordIndex== -1) wordIndex = 0; // which is UNKNOWN
			input[i][0] = wordIndex;
		}
		//System.out.println(Arrays.deepToString(input));
		return input;
	}
	
	private String mostLikelyWord(float[][] vector) {
		String word = "<ERROR>";
		System.out.println("------");
		System.out.println(Arrays.deepToString(vector));
		// find word with highest prob
		float max = vector[0][0];
		int wordIndex = 0;
		for(float e : vector[0]) 
			if (max < e && Containers.indexOf(e, vector[0]) != 0) { 
				wordIndex = Containers.indexOf(e, vector[0]);
				max = e;
			}
		word = vocab[wordIndex];
		
		return word;
	}

	/*
	 * sample a series of words from the model
	 */
	String sampleSeries(String question, int expectedAnswerLength) throws Exception {
		String answer = "<ERROR>";
		// graph obtained from running data-collection/build_graph/createLSTMGraphTF.py	
		final String graphPath = System.getProperty("user.dir") + "/data/models/final/v3/lstmGraphTF.pb";
		Path gp = Paths.get(graphPath);
		assert Files.exists(gp) : "No "+gp+" better run data-collection/build_graph/createLSTMGraphTF.py";
		final byte[] graphDef = Files.readAllBytes(gp);
		final String checkpointDir = System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + ckptVersion;
	    final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

	    try (Graph graph = new Graph();
	        Session sess = new Session(graph);
	        Tensor<String> checkpointPrefix =
	        Tensors.create(Paths.get(checkpointDir, "ckpt").toString())) {
	    	
	    	graph.importGraphDef(graphDef);
			if (checkpointExists) {						
				System.out.println("Restoring model ...");
				sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
			} else {
				//System.out.print("Error: Couldn't restore model ...");
				//return "";
				System.out.println("Initialising model ...");
				sess.runner().addTarget("init").run();
			}
			
			String[] questionArray = tokenise(question);
			// if the question is shorter than the set seq_length, we fill in the beginning slots with UNKNOWN
			// using a set seq_length is necessary for the lstm
			if(questionArray.length < seq_length) {
				String[] temp = new String[seq_length];
				Arrays.fill(temp, "START");
				System.arraycopy(questionArray, 0, temp, seq_length-questionArray.length , questionArray.length);
				questionArray = temp;
			}
			
			String[] answerArray = new String[expectedAnswerLength];
			// we generate as many words as we decided to expect the answer to have
			for (int i = 0; i < expectedAnswerLength; i++) {			
				String nextWord = "<ERROR>";
				System.out.println(Arrays.deepToString(questionArray));
				try (Tensor<?> instanceTensor = Tensors.create(wordsIntoInputVector(questionArray));
					Tensor<?> output = sess.runner().feed("input", instanceTensor).fetch("output").run().get(0).expect(Float.class)) {
					float[][] outputArray = new float[1][vocab_size];
					output.copyTo(outputArray);
					nextWord = mostLikelyWord(outputArray);
					answerArray[i] = nextWord;

					//System.out.println("-------");
					System.arraycopy(questionArray, 1, questionArray, 0, questionArray.length-1);
					questionArray[seq_length-1] = nextWord;
					//System.out.println(Arrays.deepToString(questionArray));
				}
			}
			answer = Arrays.deepToString(answerArray);
	    }
	    
		System.out.printf(
				"Instance: %s \n Prediction: %s \n",
				question, answer);
	    return answer;
		
		
//		 String modelPath = System.getProperty("user.dir") + "/data/models/final/v3/lstmGraphTF.pb";
//		 try (SavedModelBundle model = SavedModelBundle.load(modelPath, "serve")) {
//			 printSignature(model);
//			 String question = "what is probability";
//			 List<Tensor<?>> outputs = null;
//			 try (Tensor<Float> questionTensor = Tensors.create(wordsIntoFeatureVector(question))) {
//				 outputs = model
//				          .session()
//				          .runner()
//				          .feed("input", questionTensor)
//				          .fetch("output")
//				          .run();
//			}
//			try (Tensor<Float> answerTensor = outputs.get(0).expect(Float.class);) {
//				String answer = featureVectorIntoWords(answerTensor);
//			}			 
//		 }
	}
	
	/*
	 * sample a word from the model
	 */
	String sampleWord(String question) throws Exception {
		String nextWord = "<ERROR>";
		// graph obtained from running data-collection/build_graph/createLSTMGraphTF.py	
		final String graphPath = System.getProperty("user.dir") + "/data/models/final/v3/lstmGraphTF.pb";
		Path gp = Paths.get(graphPath);
		assert Files.exists(gp) : "No "+gp+" better run data-collection/build_graph/createLSTMGraphTF.py";
		final byte[] graphDef = Files.readAllBytes(gp);
		final String checkpointDir = System.getProperty("user.dir") + "/data/models/final/v3/checkpoint";
	    final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

	    try (Graph graph = new Graph();
	        Session sess = new Session(graph);
	        Tensor<String> checkpointPrefix =
	        Tensors.create(Paths.get(checkpointDir, "ckpt").toString())) {
	    	
	    	graph.importGraphDef(graphDef);
			if (checkpointExists) {						
				System.out.println("Restoring model ...");
				sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
			} else {
				System.out.print("Error: Couldn't restore model ...");
				return "";
			}
			try (Tensor<?> input = Tensors.create(wordsIntoInputVector(tokenise(question)));
					Tensor<?> output =
							sess.runner().feed("input", input).fetch("output").run().get(0).expect(Float.class)) {
				System.out.println(output.toString());
				float[][] outputArray = new float[1][vocab_size];
				output.copyTo(outputArray);
				nextWord = mostLikelyWord(outputArray);
				System.out.printf(
						"Instance: %s \n Prediction: %s \n",
						question, nextWord);
			}			
	    }
	    return nextWord;
	}
	
	
	/**
	 * 
	 * @param words
	 * @return bag-of-words 1-hot encoded, vocab_size long
	 */
	float[][] wordsIntoFeatureVector(String words) {
		// TODO there should be a more efficient way of doing this
		
		String[] splitted = tokenise(words);
		float[][] wordsOneHotEncoded = new float[1][vocab_size]; 
		Arrays.fill(wordsOneHotEncoded[0], 0);
		
		for (int i = 0; i < splitted.length; i++) {
			for (int j = 0; j < vocab.length; j++) {	
				if (vocab[j] == splitted[i]) {
					wordsOneHotEncoded[0][j] = (float) 1;
				} 
			}
		}
		
		//System.out.println(Arrays.deepToString(wordsOneHotEncoded));
		
		return wordsOneHotEncoded;
	}

	private String[] tokenise(String words) {
		String[] splitted = words.split("\\s+");
		return splitted;
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
	    System.out.printf("W = %f\tb = %f\n", values.get(0).intValue(), values.get(1).intValue());
	    for (Tensor<?> t : values) {
	      t.close();
	    }
	}
}
