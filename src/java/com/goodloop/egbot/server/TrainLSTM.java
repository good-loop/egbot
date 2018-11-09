package com.goodloop.egbot.server;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
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

import com.goodloop.egbot.EgbotConfig;
import com.sun.xml.internal.bind.v2.runtime.unmarshaller.XsiNilLoader.Array;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.json.JSONObject;
import com.winterwell.maths.stats.distributions.cond.Cntxt;
import com.winterwell.maths.stats.distributions.cond.Sitn;
import com.winterwell.nlp.corpus.SimpleDocument;
import com.winterwell.nlp.io.SitnStream;
import com.winterwell.nlp.io.Tkn;
import com.winterwell.utils.containers.Containers;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.time.RateCounter;
import com.winterwell.utils.time.TUnit;
import com.winterwell.utils.web.SimpleJson;

/**
 * @testedby {@link TrainLSTMTest}
 * @author daniel
 *
 */
public class TrainLSTM {
	// input: training data and vocab
	List<List<String>> trainingDataArray;
	HashMap<Integer, String> vocab;
	int vocab_size;

	// training parameters
	int seq_length = 3; 	// sequence length
	int num_epochs = 50000; // training epochs
	int num_hidden = 256; // number of hidden layers
	// checkpoint version to identify trained model
	int ckptVersion = new Random().nextInt(1000000);

	TrainLSTM() throws IOException{
		//trainingDataArray = loadTrainingData();
		String trainingData = "long ago , the mice had a general council to consider what measures they could take to outwit their common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal to make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in the sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her approach , we could easily escape from her . i venture , therefore , to propose that a small bell be procured , and attached by a ribbon round the neck of the cat . by this means we should always know when she was about , and could easily retire while she was in the neighbourhood . this proposal met with general applause , until an old mouse got up and said  that is all very well , but who is to bell the cat ? the mice looked at one another and nobody spoke . then the old mouse said it is easy to propose impossible remedies .";
		List<String> temp = new ArrayList<String>(Arrays.asList(tokenise(trainingData)));			
		trainingDataArray = new ArrayList<List<String>>();
		trainingDataArray.add(temp);
		//int trainingDatasize = trainingDataArray.length-1;
		initVocab();
		
		System.out.printf("Ckeckpoint no: %s\n", ckptVersion);
		System.out.printf("Training Data size: %s\n", temp.size());
		System.out.printf("Vocabulary size: %s\n", vocab_size);
		System.out.println();
	}
	
	/**
	 * load egbot zenodo files and save them in trainingDataArray as list of qa paragraphs tokenised e.g. [ [ "let", "us", "suppose", ... ] ]
	 */
	private List<List<String>> loadTrainingData() throws IOException {
		EgbotConfig config = new EgbotConfig();
		List<File> files = null;
		if (false) {
			// zenodo data slimmed down to filter only q&a body_markdown using python script data-collection/slimming.py
			// Use this for extra speed if youve run the slimming script
			// python script data-collection/slimming.py
			files = Arrays.asList(new File(config.srcDataDir, "slim").listFiles());
		} else {
			files = Arrays.asList(config.srcDataDir.listFiles(new FilenameFilter() {				
				@Override
				public boolean accept(File dir, String name) {
					return name.startsWith("MathStackExchangeAPI_Part") && name.endsWith(".json");
				}
			}));
		}
		// always have the same ordering
		Collections.sort(files);
		
		RateCounter rate = new RateCounter(TUnit.MINUTE.dt);
		
		List<List<String>> trainingData = new ArrayList<List<String>>(); 
		for(File file : files) {
			System.out.println("File: "+file+"...");
			Gson gson = new Gson();
			JsonReader jr = new JsonReader(FileUtils.getReader(file));
			jr.beginArray();
						
			int c=0;
			while(jr.hasNext()) {
				Map qa = gson.fromJson(jr, Map.class);			
				Boolean is_answered = (Boolean) qa.get("is_answered");
				if (is_answered) {
					String question_body = (String) qa.get("body_markdown");
					double answer_count = (double) qa.get("answer_count");
					for (int j = 0; j < answer_count; j++) {					
						Boolean is_accepted = (Boolean) SimpleJson.get(qa, "answers", j, "is_accepted");
						if (is_accepted) {
							String answer_body = SimpleJson.get(qa, "answers", 0, "body_markdown");
							ArrayList<String> temp = new ArrayList<String>();
							temp.add(question_body);
							temp.add(answer_body);
							trainingData.add(temp);
							c++;
							rate.plus(1);
							if (c % 1000 == 0) System.out.println(c+" "+rate+"...");
						}
					}
				}			
			} 
		}
		return trainingData;
	}

	/**
	 * initialise vocabulary
	 */
	private void initVocab(){	
		// TODO: write up vocab saving and loading (vocab_size has to be the same size in the script that constructs the graph) but how should i save it? 
		// should I use a TreeSet instead of a HashMap, log(n) for basic operations and unique https://stackoverflow.com/questions/13259535/how-to-maintain-a-unique-list-in-java
		
	    vocab = new HashMap<Integer, String>();		
	    vocab.put(0,"UNKNOWN");
		vocab.put(1,"START");
		vocab.put(2,"END");
		vocab.put(3,"ERROR");
		
		// construct vocab
		int vocabIdx = 4;
		for (int i = 0; i < trainingDataArray.size(); i++) {
			List<String> qa = trainingDataArray.get(i);
			for (int j = 0; j < qa.size(); j++) {
				if (!vocab.containsValue(qa.get(j))) {
					vocab.put(vocabIdx, qa.get(j));				
					vocabIdx += 1;
				}
			}
		}		
		vocab_size = vocab.size(); 
	}
	
	/**
	 * train the model
	 */
	public void train() throws IOException {	
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
			printVariables(sess);
			
			List<Tensor<?>> runner = null;
			float trainAccuracy = 0;
			float predAccuracy = 0;

			ArrayList<Float> accuracies = new ArrayList<Float>();
			ArrayList<Float> predictions = new ArrayList<Float>();
			
			// train a bunch of times.
			// (will be much more efficient if we sent batches instead of individual values).
			for (int epoch = 1; epoch <= num_epochs; epoch++) {
				for (int qaIdx = 0; qaIdx < trainingDataArray.size(); qaIdx++) {
					List<String> temp = trainingDataArray.get(qaIdx);
					String[] qa = temp.toArray(new String[temp.size()]);
					//System.out.println(Arrays.toString(qa.toArray(new String[qa.size()])));
					//int noOfInstances = trainingDataArray.get(qaIdx).size();
					int noOfInstances = 70;// trainingDataArray.length-seq_length;
					for (int wordIdx = 0; wordIdx < noOfInstances; wordIdx++) {
						String[] instanceArray = new String[seq_length];
						String target = "";
						Arrays.fill(instanceArray, "START");
						if (wordIdx < seq_length) {
							System.arraycopy(qa, 0, instanceArray, seq_length-wordIdx-1, wordIdx+1);
							target = qa[wordIdx+1];
						}
						else {
							// TODO: cover the case where the instance stream reached the end of the training data (aka filling in END tags)
							System.arraycopy(qa, wordIdx-seq_length+1, instanceArray, 0, seq_length);
							target = qa[wordIdx+1];
						}
//						if (epoch%100 == 0 && wordIdx == 0) {
//							System.out.printf("epoch = %d qaIdx = %d wordIdx = %d \nInstance: %s\n Target: %s \n\n", 
//									epoch, qaIdx, wordIdx, Arrays.deepToString(instanceArray), target);							
//						}

						// NB: try-with to ensure C level resources are released
						try (Tensor<?> instanceTensor = Tensors.create(wordsIntoInputVector(instanceArray));
								Tensor<?> targetTensor = Tensors.create(wordsIntoFeatureVector(target))) {
							// The names of the tensors and operations in the graph are printed out by the program that created the graph
					    	// you can find the names in the following file: data/models/final/v3/tensorNames.txt
							runner = sess.runner()
									.feed("input", instanceTensor)
									.feed("target", targetTensor)
									.addTarget("train_op")
									.fetch("correct_pred")
									.fetch("accuracy")
									.run();
							
//							boolean[] bArray = new boolean[1];
//		 					try(Tensor correct_pred = runner.get(0)){
//								correct_pred.copyTo(bArray);
////							 	if(bArray[0]) {
////							 		predictions.add((float)1);
////							 	}
////							 	else {
////							 		predictions.add((float)0);
////							 	}
//		 					}
							float acc = runner.get(1).floatValue();
							accuracies.add(acc);
							closeTensors(runner);
						}
					}
				}
				String nextWord = "";
				if (epoch%100 == 0) {				
					// inference 
					String[] testInstance = new String[seq_length];
					List<String> temp = trainingDataArray.get(0);
					String[] tempArray = temp.toArray(new String[temp.size()]);
					int startIdx = new Random().nextInt(temp.size()-seq_length);
					System.arraycopy(tempArray, startIdx, testInstance, 0, seq_length);
					try (Tensor<?> input = Tensors.create(wordsIntoInputVector(testInstance));
						Tensor<?> target = Tensors.create(wordsIntoFeatureVector(tempArray[startIdx+seq_length]))) {
						List<Tensor<?>> outputs = sess.runner()
								.feed("input", input)
								.feed("target", target)
								.fetch("output")
								.fetch("correct_pred")
								.fetch("accuracy")
								.run();
						
						float[][] outputArray = new float[1][vocab_size];
						outputs.get(0).copyTo(outputArray);
						nextWord = mostLikelyWord(outputArray);
						
						boolean[] bArray = new boolean[1];
	 					try(Tensor correct_pred = outputs.get(1)){
							correct_pred.copyTo(bArray);
	 					}
						if(bArray[0]) {
					 		predictions.add((float)1);
					 	}
					 	else {
					 		predictions.add((float)0);
					 	}
											
						float sum = 0;
						for (float i : predictions)
						    sum = sum + i;		 				 	
						predAccuracy = sum / predictions.size() * 100;
						
						sum = 0;
						for (float i : accuracies)
						    sum = sum + i;
						trainAccuracy = sum / accuracies.size() * 100;

						closeTensors(outputs);
					}    
					if (epoch%1000 == 0) {
						printVariables(sess);
						System.out.printf("\nAfter %d examples: \n", epoch*trainingDataArray.size());
						System.out.printf("Correct pred: %f \n", predAccuracy);
						System.out.printf("Accuracy: %f \n", trainAccuracy);
						
						System.out.printf(
								" Instance: %s \n Prediction: %s \n Target: %s \n",
								Arrays.deepToString(testInstance), nextWord, tempArray[startIdx+seq_length]);
					}
				}
			}

			// checkpoint
			sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();
	    }
	}

	/*
	 * close tensors to release memory used by them (to be used when try catch statement can't be used)
	 */
	private static void closeTensors(final Collection<Tensor<?>> ts) {      
	    for (final Tensor<?> t : ts) {
	        try {
	            t.close();
	        } catch (final Exception e) {
	            System.err.println("Error closing Tensor.");
	            e.printStackTrace();
	        }
	    }
	    ts.clear();
	}
	
	/**
	 * 
	 * @param testInstance
	 * @return e.g. [[7], [13], [12] ...] 30 long  to match seq_length=30
	 */
	private float[][] wordsIntoInputVector(String[] words) {
		//System.out.println("\n");
		//System.out.println(testInstance);		
		float[][] input = new float[seq_length][1];
		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			int wordIndex = 0; // index 0 represents UNKNOWN
			if (vocab.containsValue(word)) wordIndex = getKeyByValue(vocab, word); 
			input[i][0] = wordIndex;
		}
		//System.out.println(Arrays.deepToString(input));
		return input;
	}
	
	/**
	 * get key based on value in a HashMap
	 */
	private static <T, E> int getKeyByValue(Map<T, E> map, E value) {
	    for (Entry<T, E> entry : map.entrySet()) {
	        if (Objects.equals(value, entry.getValue())) {
	            return (int) entry.getKey();
	        }
	    }
	    return 0;
	}
	
	private String mostLikelyWord(float[][] vector) {
		String word = "<ERROR>";
		String[] vocabArray = vocab.values().toArray(new String[0]);
	    //System.out.printf("Vocabulary:\t %s\n", Arrays.toString(vocabArray));
	    //System.out.printf("Probabilities:\t %s\n", Arrays.toString(vector[0]));
		// find word with highest prob
		float max = vector[0][0];
		int wordIndex = 0;
		for(float e : vector[0]) 
			if (max < e && Containers.indexOf(e, vector[0]) != 0) { 
				wordIndex = Containers.indexOf(e, vector[0]);
				max = e;
			}
		if (wordIndex!=0) {
			word = vocab.get(wordIndex);
		}
		return word;
	}
	
	/**
	 * 
	 * @param words
	 * @return bag-of-words 1-hot encoded, vocab_size long
	 */
	private float[][] wordsIntoFeatureVector(String words) {
		// TODO there should be a more efficient way of doing this
		
		String[] splitted = tokenise(words);
		float[][] wordsOneHotEncoded = new float[1][vocab_size]; 
		Arrays.fill(wordsOneHotEncoded[0], 0);
		
		int rem = -1;
		// don't we always expect just one word? 
		for (int i = 0; i < splitted.length; i++) {
			if (vocab.containsValue(splitted[i])) {
				int vocabIdx = getKeyByValue(vocab, splitted[i]);
				wordsOneHotEncoded[0][vocabIdx] = (float) 1;
				rem = vocabIdx;
			}
			else {
				System.out.printf("Couldn't find word in vocab: %s\n", words);
				System.exit(0);
			}
		}
//		System.out.println(words);
//		System.out.println(Arrays.deepToString(wordsOneHotEncoded));
//		System.out.printf("Idx: %d\n", rem);

		return wordsOneHotEncoded;
	}

	private String[] tokenise(String words) {
		String[] splitted = words.split("\\s+");
		return splitted;
	}

	/*
	 * sample a series of words from the model
	 */
	public String sampleSeries(String question, int expectedAnswerLength) throws Exception {
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
				System.out.print("Error: Couldn't restore model ...");
				return "";
			}
			
			String[] questionArray = tokenise(question);
			// if the question is shorter than the set seq_length, we fill in the beginning slots with START
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
	public String sampleWord(String question) throws Exception {
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
	
	
	private String featureVectorIntoWords(Tensor<?> answerTensor) {
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
	
	private void printVariables(Session sess) {
	    List<Tensor<?>> values = sess.runner().fetch("W/read").fetch("b/read").run();
	    float[][] w = new float[num_hidden*2][vocab_size]; // 2*num_hidden because of forward + backward cells
	    values.get(0).copyTo(w);
	    float[] b = new float[vocab_size];
	    values.get(1).copyTo(b);
	    System.out.printf("W0 = %s\nb = %s\n\n", Arrays.toString(w[0]), Arrays.toString(b));
	    for (Tensor<?> t : values) {
	      t.close();
	    }
	}
}
