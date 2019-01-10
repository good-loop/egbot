package com.goodloop.egbot.server;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.TensorFlow;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.GPUOptions;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.types.UInt8;

import com.goodloop.egbot.EgbotConfig;
import com.sun.xml.internal.bind.v2.runtime.unmarshaller.XsiNilLoader.Array;
import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.json.JSONObject;
import com.winterwell.maths.datastorage.HalfLifeMap;
import com.winterwell.maths.stats.distributions.cond.Cntxt;
import com.winterwell.maths.stats.distributions.cond.ICondDistribution;
import com.winterwell.maths.stats.distributions.cond.Sitn;
import com.winterwell.maths.stats.distributions.discrete.IFiniteDistribution;
import com.winterwell.nlp.NLPWorkshop;
import com.winterwell.nlp.corpus.SimpleDocument;
import com.winterwell.nlp.dict.Dictionary;
import com.winterwell.nlp.dict.EmoticonDictionary;
import com.winterwell.nlp.io.FilteredTokenStream;
import com.winterwell.nlp.io.ITokenStream;
import com.winterwell.nlp.io.SitnStream;
import com.winterwell.nlp.io.StopWordFilter;
import com.winterwell.nlp.io.Tkn;
import com.winterwell.nlp.io.WordAndPunctuationTokeniser;
import com.winterwell.nlp.io.FilteredTokenStream.KInOut;
import com.winterwell.utils.Utils;
import com.winterwell.utils.containers.Containers;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.KErrorPolicy;
import com.winterwell.utils.log.Log;
import com.winterwell.utils.time.RateCounter;
import com.winterwell.utils.time.TUnit;
import com.winterwell.utils.web.SimpleJson;

/**
 * @testedby {@link TrainLSTMTest}
 * @author daniel
 *
 */
public class LSTM implements IEgBotModel {
	// model guts
	List<Tensor<?>> model;
	public Desc desc;
	
	// true when model was succesfully loaded
	public boolean loadSuccessFlag;

	// true once training finished on all egbot files
	public boolean trainSuccessFlag; 

	// vocab
	HashMap<Integer, String> vocab;	
	int idealVocabSize;
	int vocab_size;
	
	// training parameters
	int seq_length; // sequence length
	int num_epochs; // training epochs 
	int num_hidden; // number of hidden layers
	
	// output location
	String saveLocation;
	String backupLocation;
	String logLocation;
	
	// stats tracker
	ArrayList<Float> trainAccuracies;
	int sessRunCount;
	int questionCount;
	long lStartTime;
	

	/**
	 * default constructor
	 * @throws IOException
	 */
	LSTM() throws IOException{
		model = new ArrayList<Tensor<?>>();
		desc = new Desc<>("MSE-lstm-slim", model.getClass());
		//String saveLocation = "/data/models/final/v3/checkpoint" + desc.getName();
		logLocation = "/home/irina/egbot-learning-depot/results/log.txt";
		saveLocation = "/home/irina/egbot-learning-depot/results/" + desc.getName();
		backupLocation = "/home/irina/egbot-learning-depot/backups";
		trainAccuracies = new ArrayList<Float>();
	}
	
	/**
	 * initialise any model parameters to prepare for training
	 */
	public void init(List<File> files) throws IOException {
		// training parameters
		seq_length = 30; 	// sequence length
		num_epochs = 10; // training epochs 
		num_hidden = 256; // number of hidden layers
		idealVocabSize = 10000;
		sessRunCount = 0;
		questionCount = 0;
		lStartTime = System.nanoTime();	

		loadVocab(files);
	}
	
	/**
	 * load egbot slim files and construct vocab (without saving training data because it's too memory consuming)
	 * 
	 * This does NOT init the vocab field. Use {@link #loadVocab(List)} to load
	 * 
	 * @return magic version number -- needed to load
	 */
	public void loadVocab2_trainVocabAndSave(List<File> files) throws IOException {
		// vocab has to be constructed and saved from all the text that will be used when training 
		// this is because vocab_size defines the shape of the feature vectors
		System.out.println("Loading files and initialising vocabulary");
		 	
		// construct vocab that auto-prunes and discards words that appear rarely
		// hlVocab is a map where the key to be the word and the value to be the word counts
		HalfLifeMap<String, Integer> hlVocab = new HalfLifeMap<String, Integer>(idealVocabSize);
		
		RateCounter rate = new RateCounter(TUnit.MINUTE.dt);
		
		for(File file : files) {
			System.out.println("File: "+file+"...");
			Gson gson = new Gson();
			JsonReader jr = new JsonReader(FileUtils.getReader(file));
			jr.beginArray();
						
			int c=0;
			while(jr.hasNext()) {
				Map qa = gson.fromJson(jr, Map.class);			
				String question_body = (String) qa.get("question");
				String answer_body = (String) qa.get("answer");
				String[] temp = EgBotDataLoader.tokenise(question_body + " " + answer_body);
				for (String word : temp) {
					if (word.isEmpty()) continue;
					Integer cnt = hlVocab.get(word);
					if (cnt!=null) {
						int count = cnt + 1;
						hlVocab.put(word, count);
					} else {
						hlVocab.put(word, 1);
					}		
				}				
				c++;
				rate.plus(1);
				if (c % 1000 == 0) System.out.println("Count: "+c+"\t Rate: "+rate+"\t Vocab size: "+hlVocab.size());
			}
			jr.close();
		}
		
		// save to file
		Set<String> vocabWords = hlVocab.keySet();
		
		File vocabFile = vocabPath();
		vocabFile.createNewFile();		
		try (PrintWriter out = new PrintWriter(vocabFile)) {			
			for (String word : vocabWords) {
			    out.println(word);
			}
		}		
		System.out.printf("Saved vocab to file: %s\n", vocabFile);
		System.out.printf("Initialised vocabulary size: %s\n", hlVocab.size());
		
		// find out top X vocab words
		//vocabTop(100,hlVocab);
	}
	
	/**
	 * top SIZE words in the vocabulary (where size could be 1000 for the 1000th most common words)
	 * can only be called from {@link #loadVocab2_trainVocabAndSave} 
	 * because we don't keep the vocab counts once we're done building the vocab
	 * 
	 */
	public ArrayList<String> vocabTop(int size, HalfLifeMap<String, Integer> hlVocab) throws IOException {
		ArrayList<String> topArray = new ArrayList<String>(size);
		
		List<String> keysSortedByValue = Containers.getSortedKeys(hlVocab);
		Collections.reverse(keysSortedByValue); // largest first
				
		for (String s : keysSortedByValue) {
			//ITokenStream a = null;
			System.out.println(s);
			topArray.add(s);
			if (topArray.size() == size) break;
		}
		return topArray;
	}
	
	/**
	 * method to estimate how long training would take
	 * @throws IOException
	 */
	public void checkTrainSize(List<File> files) throws IOException {
		RateCounter rate = new RateCounter(TUnit.MINUTE.dt);
		List<List<String>> trainingBatch = new ArrayList<List<String>>(); 
	
		int c = 0;
		for(File file : files) {
			System.out.println("File: "+file+"...");
			Gson gson = new Gson();
			JsonReader jr = new JsonReader(FileUtils.getReader(file));
			jr.beginArray();
						
			while(jr.hasNext()) c++;
		}
		System.out.printf("No of training items: %d", c);
	}
	
	/**
	 * load egbot zenodo files, save content locally in trainingDataArray as list of qa paragraphs tokenised e.g. [ [ "let", "us", "suppose", ... ] ] and then run train for each file 
	 * @throws IOException 
	 */
	@Deprecated
	public void train(List<File> files) throws IOException {
		// load files, save content locally and train using local data and loaded vocab from file
		RateCounter rate = new RateCounter(TUnit.MINUTE.dt);
		List<List<String>> trainingBatch = new ArrayList<List<String>>(); 


		for(File file : files) {
			System.out.println("File: "+file+"...");
			Gson gson = new Gson();
			JsonReader jr = new JsonReader(FileUtils.getReader(file));
			jr.beginArray();
						
			int c=0;
			while(jr.hasNext() && c < 50) { // TODO: remove 2nd condition; it's a temporary limitation to produce dummy trained lstm model;
				Map qa = gson.fromJson(jr, Map.class);			
				String question_body = (String) qa.get("question");
				String answer_body = (String) qa.get("answer");
				String[] temp = EgBotDataLoader.tokenise(question_body + " " + answer_body);
				trainingBatch.add(Arrays.asList(temp));
				trainEach(trainingBatch);
				c++;
				rate.plus(1);
				// TODO: mini-batch training?
//				if (c % 10 == 0) {
//					System.out.println(c+" "+rate+"...");
//					trainEach(trainingBatch);
//					trainingBatch = new ArrayList<List<String>>(); 
//				}
//				if (c % 1000 == 0) {
//					//train in batches to prevent memory issues
//					trainEach(trainingBatch);
//					trainingBatch = new ArrayList<List<String>>(); 
//					System.out.println(c+" "+rate+"...");
//				}	
			}			
			// close file to save memory
			jr.close();

			System.out.printf("Saved trained model to file: %s\n", saveLocation);
		}
	}

	/**
	 * load vocabulary from file, adding 4 special tokens at the start
	 * @param string see {@link #loadAndInitVocab()}
	 * @throws IOException
	 */
	public void loadVocab(List<File> files) throws IOException{	
		
		// load the vocab to a map that allows for unique indexing of words
		// vocab is a map where the key is the unique index of the word, and the value is the word itself
	    vocab = new HashMap<Integer, String>();		
	    vocab.put(0,"UNKNOWN");
		vocab.put(1,"START");
		vocab.put(2,"END");
		vocab.put(3,"ERROR");
		int vocabIdx = 4;
		
		File vocabFile = vocabPath();
        // checks to see if it finds the vocab file
	    final boolean checkpointExists = vocabFile.exists();
	    if( ! checkpointExists) {
	    	Log.d("Warning: Couldn't find vocab file");
	    	// train and save to file
	    	// NB: we then promptly load from file below, which is a tiny bit inefficient, but tiny. 
	    	loadVocab2_trainVocabAndSave(files); 
	    }
        try(BufferedReader br = new BufferedReader(new FileReader(vocabFile))) {
            for(String word; (word = br.readLine()) != null; ) {
    			vocab.put(vocabIdx, word);
    			vocabIdx += 1;
            }
        }
		vocab_size = vocab.size();
		System.out.printf("Loaded vocabulary size: %s\n", vocab_size);
	}

	private File vocabPath() {							
		String vocabPath = 	System.getProperty("user.dir") + "/data/models/final/v3/vocab_" + desc.getName()  + ".txt";
		return new File(vocabPath);
	}
	
	/**
	 * train the model and save it in /data/models/final/v3/checkpoint<VERSION_NUMBER>
	 * @param trainingDataArray is the data that will be used for training, expected to be in the shape of a list of qa paragraphs tokenised e.g. [ [ "let", "us", "suppose", ... ] ] 
	 * @throws IOException
	 */
	public void trainEach(List<List<String>> trainingDataArray) throws IOException {	
		
		// graph obtained from running data-collection/build_graph/createLSTMGraphTF.py	
		final String graphPath = System.getProperty("user.dir") + "/data/models/final/v3/lstmGraphTF.pb";
		Path gp = Paths.get(graphPath);
		assert Files.exists(gp) : "No "+gp+" better run data-collection/build_graph/createLSTMGraphTF.py";
		final byte[] graphDef = Files.readAllBytes(gp);
		final String checkpointDir = saveLocation; //System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + desc.getName();
	    final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

	    // setting session config to use the GPU
        GPUOptions gpuOptions = GPUOptions.newBuilder()
        		.setPerProcessGpuMemoryFraction(1)
                .setForceGpuCompatible(true)
                .setAllowGrowth(true)
                .build();
        
        ConfigProto config = ConfigProto.newBuilder()
        		//.setLogDevicePlacement(true)
                .setGpuOptions(gpuOptions)
                .build();

	    // load graph
	    try (Graph graph = new Graph();
	        Tensor<String> checkpointPrefix =
	        Tensors.create(Paths.get(checkpointDir, "ckpt").toString())) {
	    		    		    	
	    	graph.importGraphDef(graphDef); //builder.build().toByteArray());
	    	try(Session sess = new Session(graph, config.toByteArray())){
		    	// initialise or restore.
				// The names of the tensors and operations in the graph are printed out by the program that created the graph
		    	// you can find the names in the following file: data/models/final/v3/tensorNames.txt
				if (checkpointExists) {						
					//System.out.println("Restoring model ...");
					sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
				} else {
					System.out.println("Initialising model ...");
					sess.runner().addTarget("init").run();
				}
				
				// print out weight and bias initialisation
				//System.out.print("Starting from: \n");
				//printVariables(sess);
				
				questionCount ++;
				
				// train a bunch of times
				// TODO: will it be more efficient if we sent batches instead of individual values?
				for (int epoch = 1; epoch <= num_epochs; epoch++) {				
					trainEach2_epoch(trainingDataArray, sess, trainAccuracies, epoch, questionCount);
				}
	
				// save model checkpoint
				save(sess,checkpointPrefix);
	    	}
	    }
	}

	private void trainEach2_epoch(List<List<String>> trainingDataArray, Session sess, ArrayList<Float> trainAccuracies, int epoch, int questionCount) 
	{		
		// for each qa segment
		for (int batchIdx = 0; batchIdx < trainingDataArray.size(); batchIdx++) {
			// A Q+A training string
			List<String> temp = trainingDataArray.get(batchIdx);
			String[] qa = temp.toArray(new String[temp.size()]);
			
			// for each word in the qa segment (not including the last seq_length ones)
			// TODO: cover the case where the instance stream reached the end of the training data (aka filling in END tags)?
			for (int wordIdx = 0; wordIdx < qa.length-seq_length; wordIdx++) {
				String[] instanceArray = new String[seq_length];
				String target = "";
				// filling in seq_length-1 START tags to allow guessing of words at the beginning (aka where the word position < seq_length)
				Arrays.fill(instanceArray, "START");
				if (wordIdx < seq_length) {
					System.arraycopy(qa, 0, instanceArray, seq_length-wordIdx-1, wordIdx+1);
				}
				else {
					System.arraycopy(qa, wordIdx-seq_length+1, instanceArray, 0, seq_length);
				}
				target = qa[wordIdx+1];
				
				// print out training example and some stats occasionally
				if (sessRunCount%10000 == 1) {
					System.out.printf("sessRunCount = %d batchIdx = %d qIdx = %d epoch = %d wordIdx = %d \nInstance: %s\n Target: %s \n\n", 
							sessRunCount, batchIdx, questionCount, epoch, wordIdx, Arrays.deepToString(instanceArray), target);
					
			        long lEndTime = System.nanoTime();
			        long output = lEndTime - lStartTime;
			        System.out.println("No of questions processed: " + questionCount);
			        System.out.println("Elapsed time in seconds: " + output / 1000000000);
			        System.out.println("Rate (training iter/ sec): " + (float)sessRunCount/ ((float)output / 1000000000));
				}

				// create input and output tensors and run training operation
				// NB: try-with to ensure C level resources are released
				try (Tensor<?> instanceTensor = Tensors.create(wordsIntoInputVector(instanceArray));
						Tensor<?> targetTensor = Tensors.create(wordsIntoFeatureVector(target))) {
					// The names of the tensors and operations in the graph are printed out by the program that created the graph
			    	// you can find the names in the following file: data/models/final/v3/tensorNames.txt
					List<Tensor<?>> runner = sess.runner()
							.feed("input", instanceTensor)
							.feed("target", targetTensor)
							.addTarget("train_op")
							.fetch("accuracy") // training accuracy
							.run();
					sessRunCount++;
					
					trainAccuracies.add(runner.get(0).floatValue());							

					// close tensors to save memory
					closeTensors(runner);
				}
			} // ./ each word
		} // ./ each q-a example
	}

	/**
	 * saves trained model in the standard tensorflow checkpoint format
	 * @param sess
	 * @param checkpointPrefix
	 * @throws IOException 
	 */
	private void save(Session sess, Tensor<?> checkpointPrefix) throws IOException {
		// save latest model 
		model = sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();
		// do backup every so often (for the full 500,000 questions, there should then be 50 backups)
		if(questionCount%10000==0) {
			backupLocation = backupLocation + "/backup" + questionCount/10000;
			new File(backupLocation).mkdir();
			try(Tensor<String> backupPrefix =
			        Tensors.create(Paths.get(backupLocation, "ckpt").toString())){
			model = sess.runner().feed("save/Const", backupPrefix).addTarget("save/control_dependency").run();
			}
		}
		// save stats log
        long lEndTime = System.nanoTime();
        long output = lEndTime - lStartTime;
        long sumAcc = 0;
        for(Float acc : trainAccuracies)
        	sumAcc += acc;
        	
		try(FileWriter fw = new FileWriter(logLocation, true);
			    BufferedWriter bw = new BufferedWriter(fw);
			    PrintWriter out = new PrintWriter(bw)) {
			out.println(desc);
			out.println("No of questions processed: " + questionCount);
			out.println("Elapsed time in seconds: " + output / 1000000000);
	        out.println("Rate (training iter/ sec): " + (float)sessRunCount/ ((float)output / 1000000000));
	        out.println("Avg train accuracy: " + (float)sumAcc/sessRunCount);
	        out.println();
			out.close();
		}
	}

	/**
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
	 * convert an array of strings to an input vector (word => word's position in the vocab) 
	 * @param words 
	 * @return e.g. [[7], [13], [12] ...] that is seq_length long
	 */
	private float[][] wordsIntoInputVector(String[] words) {
		float[][] input = new float[seq_length][1];
		for (int i = 0; i < words.length; i++) {
			if (i == seq_length) break; 
			String word = words[i];
			int wordIndex = 0; // index 0 represents <UNKNOWN>
			if (vocab.containsValue(word)) wordIndex = getKeyByValue(vocab, word); 
			input[i][0] = wordIndex;
		}
		return input;
	}
	
	/**
	 * get key based on value in a HashMap (used to get the vocab position when given a word)
	 */
	private static <T, E> int getKeyByValue(Map<T, E> map, E value) {
	    for (Entry<T, E> entry : map.entrySet()) {
	        if (Objects.equals(value, entry.getValue())) {
	            return (int) entry.getKey();
	        }
	    }
	    return 0;
	}
	
	/**
	 * finds most likely next word by looking for the vector position with the biggest probability and getting the word that corresponds to that vector position
	 * @param vector
	 * @return
	 */
	private String mostLikelyWord(float[][] vector) {
		String word = "<ERROR>";
		String[] vocabArray = vocab.values().toArray(new String[0]);

		// find word with highest probability
		float max = vector[0][0];
		int wordIndex = 0;
		for(float e : vector[0]) 
			if (max < e && Containers.indexOf(e, vector[0]) != 0) { 
				wordIndex = Containers.indexOf(e, vector[0]);
				max = e;
			}
		if (wordIndex!=0) { // if it's not <UNKNOWN>
			word = vocab.get(wordIndex); // get the word from vocab
		}
		return word;
	}
	
	/**
	 * converts a string to a feature vector
	 * @param words
	 * @return bag-of-words 1-hot encoded, vocab_size long
	 */
	private float[][] wordsIntoFeatureVector(String words) {
		// TODO there should be a more efficient way of doing this
		
		String[] splitted = EgBotDataLoader.tokenise(words);
		float[][] wordsOneHotEncoded = new float[1][vocab_size]; 
		Arrays.fill(wordsOneHotEncoded[0], 0);
		
		int rem = -1;
		// don't we always expect just one word? if so, we should make it clearer
		for (int i = 0; i < splitted.length; i++) {
			if (vocab.containsValue(splitted[i])) {
				int vocabIdx = getKeyByValue(vocab, splitted[i]);
				wordsOneHotEncoded[0][vocabIdx] = (float) 1;
				rem = vocabIdx;
			}
			else {
				//System.out.printf("Couldn't find word in vocab: %s\n", words);
			}
		}
		return wordsOneHotEncoded;
	}
	
	/**
	 * sample a series of words from the model
	 * @param question
	 * @param expectedAnswerLength
	 * @return answer
	 * @throws IOException 
	 */
	public String sample(String question, int expectedAnswerLength) throws IOException {
		String answer = "<ERROR>";
		
		// graph obtained from running data-collection/build_graph/createLSTMGraphTF.py	
		final String graphPath = System.getProperty("user.dir") + "/data/models/final/v3/lstmGraphTF.pb";
		Path gp = Paths.get(graphPath);
		assert Files.exists(gp) : "No "+gp+" better run data-collection/build_graph/createLSTMGraphTF.py";
		final byte[] graphDef = Files.readAllBytes(gp);
		final String checkpointDir = saveLocation;//System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + desc.getName();
	    final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

	    // load graph
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
				System.out.println("Couldn't find model ...\n");
				return "";
			}
				        
			String[] questionArray = EgBotDataLoader.tokenise(question);
			
			// if the question is shorter than the set seq_length, we fill in the beginning slots with START (using a set seq_length is necessary for the lstm)
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
				
				// run graph with given input and fetch output
				try (Tensor<?> input = Tensors.create(wordsIntoInputVector(questionArray))) {
					List<Tensor<?>> outputs = sess.runner()
							.feed("input", input)
							.fetch("output")
							.run();
					
					// copy tensor to array
					float[][] outputArray = new float[1][vocab_size];
					outputs.get(0).copyTo(outputArray);
					nextWord = mostLikelyWord(outputArray);
					
					// add generated word to answer 
					answerArray[i] = nextWord;
					
					//shift questionArray to include the generated word at the end so as to allow us to generate the next word after that
					System.arraycopy(questionArray, 1, questionArray, 0, questionArray.length-1);
					questionArray[seq_length-1] = nextWord;
				}
			}
			answer = Arrays.deepToString(answerArray);
	    }
	    
//		System.out.printf(
//				"Instance: %s \n Prediction: %s \n",
//				question, answer);
	    return answer;
	}
	
	/**
	 * sample a word from the model
	 * @param question
	 * @return nextWord
	 * @throws Exception
	 */
	public String sampleWord(String question) throws Exception {
		String nextWord = "<ERROR>";
		
		// graph obtained from running data-collection/build_graph/createLSTMGraphTF.py	
		final String graphPath = System.getProperty("user.dir") + "/data/models/final/v3/lstmGraphTF.pb";
		
		Path gp = Paths.get(graphPath);
		assert Files.exists(gp) : "No "+gp+" better run data-collection/build_graph/createLSTMGraphTF.py";
		final byte[] graphDef = Files.readAllBytes(gp);
		final String checkpointDir = saveLocation;//System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + desc.getName();
	    final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

	    // load graph
	    try (Graph graph = new Graph();
	        Session sess = new Session(graph);
	        Tensor<String> checkpointPrefix =
	        Tensors.create(Paths.get(checkpointDir, "ckpt").toString())) {
	    	
	    	graph.importGraphDef(graphDef);
			if (checkpointExists) {						
				System.out.println("Restoring model ...");
				sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
			} else {
				System.out.print("Error: Couldn't restore model ...\n");
				return "";
			}
			
			// run graph with given input and fetch output
			try (Tensor<?> input = Tensors.create(wordsIntoInputVector(EgBotDataLoader.tokenise(question)));
					Tensor<?> output =
							sess.runner()
							.feed("input", input)
							.fetch("output")
							.run().get(0)
							.expect(Float.class)) {
				
				// copy tensor to array
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
	 * score answer for given question (where the score is the avg log probability of each word in the answer being predicted)
	 * @param q question
	 * @param t target answer
	 * @return
	 * @throws IOException 
	 */
	public double scoreAnswer(String q, String t) throws IOException {
		double score = 0; 

		// graph obtained from running data-collection/build_graph/createLSTMGraphTF.py	
		final String graphPath = System.getProperty("user.dir") + "/data/models/final/v3/lstmGraphTF.pb";
		
		Path gp = Paths.get(graphPath);
		assert Files.exists(gp) : "No "+gp+" better run data-collection/build_graph/createLSTMGraphTF.py";
		final byte[] graphDef = Files.readAllBytes(gp);
		final String checkpointDir = saveLocation;//System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + desc.getName();
	    final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

	    // load graph
	    try (Graph graph = new Graph();
	        Session sess = new Session(graph);
	        Tensor<String> checkpointPrefix =
	        Tensors.create(Paths.get(checkpointDir, "ckpt").toString())) {
	    	
	    	graph.importGraphDef(graphDef);
			if (checkpointExists) {						
				//System.out.println("Restoring model ...");
				sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
			} else {
				System.out.print("Error: Couldn't restore model ...\n");
				return 0;
			}
			
			// tokenise input
			String[] qArray = EgBotDataLoader.tokenise(q);
			String[] tArray = EgBotDataLoader.tokenise(t);
			
			// ensure we take the last 30 words of the question (or if it's < 30, then fill it with <START> tags at the beginning)
			String[] instanceArray = new String[seq_length];
			Arrays.fill(instanceArray, "START");
			if (qArray.length < seq_length) {
				System.arraycopy(qArray, 0, instanceArray, seq_length-qArray.length, qArray.length);
			}
			else {
				System.arraycopy(qArray, qArray.length-seq_length, instanceArray, 0, seq_length);
			}
			
			// for each target word
			for (int i = 0; i < tArray.length; i++) {
				try (Tensor<?> input = Tensors.create(wordsIntoInputVector(instanceArray));
					Tensor<?> target = Tensors.create(wordsIntoFeatureVector(tArray[i]))) {
					List<Tensor<?>> outputs = sess.runner()
							.feed("input", input)
							.feed("target", target)
							.fetch("output") // prediction
							.fetch("correct_pred") // validation accuracy
							.run();
					
					// copy output tensors to array
					float[][] outputArray = new float[1][vocab_size];
					outputs.get(0).copyTo(outputArray);
					// find out the position of the target word in the vocab
					int targetPos = getKeyByValue(vocab,tArray[i]);
					// find out the prob of that word, as returned from the model
					float probWord = outputArray[0][targetPos];
					// add the log prob to the score
					score += Math.log(probWord);
				}
				
				//shift questionArray to include the next targer word at the end so as to allow us to generate the next word after that
				System.arraycopy(instanceArray, 1, instanceArray, 0, instanceArray.length-1);
				instanceArray[seq_length-1] = tArray[i];
			}
			// avg the score and then return it
			return score/tArray.length;
	    }		
	}	
	
	/**
	 * Prints out the signature of the model, which specifies what type of model is being exported, 
	 * and the input/output tensors to bind to when running inference.
	 * @param model
	 * @throws Exception
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
	
	/**
	 * prints out the weights and biases, for debugging purposes 
	 * @param sess
	 */
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

	/**
	 * method called by EgBotDataLoader to train on new data point
	 * this method is separate from trainEach, in case we want to do mini-batch training 
	 */
	@Override
	public void train1(Map qa) {
		String question_body = (String) qa.get("question");
		String answer_body = (String) qa.get("answer");
		// Simple combine of Q+A
		String q_a = question_body + " " + answer_body;
		// tokenise
		String[] temp = EgBotDataLoader.tokenise(q_a);
		
		// only uses a batch of one for now
		List<List<String>> trainingBatch = new ArrayList(); 
		trainingBatch.add(Arrays.asList(temp));
		try {
			trainEach(trainingBatch);
		} catch (IOException e) {
			throw Utils.runtime(e);
		} 				
	}

	/**
	 * load saved model
	 */
	public void load() throws IOException {
		// graph obtained from running data-collection/build_graph/createLSTMGraphTF.py	
		final String graphPath = System.getProperty("user.dir") + "/data/models/final/v3/lstmGraphTF.pb";
		Path gp = Paths.get(graphPath);
		assert Files.exists(gp) : "No "+gp+" better run data-collection/build_graph/createLSTMGraphTF.py";
		final byte[] graphDef = Files.readAllBytes(gp);
		final String checkpointDir = saveLocation;//System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + desc.getName();
	    final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

	    // load graph
	    try (Graph graph = new Graph();
	        Session sess = new Session(graph);
	        Tensor<String> checkpointPrefix =
	        Tensors.create(Paths.get(checkpointDir, "ckpt").toString())) {
	    	
	    	graph.importGraphDef(graphDef);
	    	// initialise or restore.
			// The names of the tensors and operations in the graph are printed out by the program that created the graph
	    	// you can find the names in the following file: data/models/final/v3/tensorNames.txt
			if (checkpointExists) {						
				//System.out.println("Restoring model ...");
				sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/restore_all").run();
				loadSuccessFlag = true;
			} else {
				System.out.println("Initialising model ...");
				sess.runner().addTarget("init").run();
			}
	    }
	}
	
	/**
	 * Get the most likely series of words from the model.
	 *  
	 * @param question
	 * @param expectedAnswerLength
	 * @return answer
	 * @throws Exception
	 */
	public String generateMostLikely(String question, int expectedAnswerLength) throws IOException {
		String answer = "<ERROR>";
		
		// graph obtained from running data-collection/build_graph/createLSTMGraphTF.py	
		final String graphPath = System.getProperty("user.dir") + "/data/models/final/v3/lstmGraphTF.pb";
		Path gp = Paths.get(graphPath);
		assert Files.exists(gp) : "No "+gp+" better run data-collection/build_graph/createLSTMGraphTF.py";
		final byte[] graphDef = Files.readAllBytes(gp);
		final String checkpointDir = saveLocation;//System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + desc.getName();
	    final boolean checkpointExists = Files.exists(Paths.get(checkpointDir));

	    // load graph
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
				System.out.println("Couldn't find model ...\n");
				return "";
			}
			
			String[] questionArray = EgBotDataLoader.tokenise(question);
			
			// if the question is shorter than the set seq_length, we fill in the beginning slots with START (using a set seq_length is necessary for the lstm)
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
				
				// run graph with given input and fetch output
				try (Tensor<?> input = Tensors.create(wordsIntoInputVector(questionArray))) {
					List<Tensor<?>> outputs = sess.runner()
							.feed("input", input)
							.fetch("output")
							.run();
					
					// copy tensor to array
					float[][] outputArray = new float[1][vocab_size];
					outputs.get(0).copyTo(outputArray);
					nextWord = mostLikelyWord(outputArray);
					
					// add generated word to answer 
					answerArray[i] = nextWord;
					
					//shift questionArray to include the generated word at the end so as to allow us to generate the next word after that
					System.arraycopy(questionArray, 1, questionArray, 0, questionArray.length-1);
					questionArray[seq_length-1] = nextWord;
				}
			}
			answer = Arrays.deepToString(answerArray);
	    }
	    
	    return answer;
	}
	
	public boolean isLoadSuccessFlag() {
		return loadSuccessFlag;
	}

	public void setLoadSuccessFlag(boolean loadSuccessFlag) {
		this.loadSuccessFlag = loadSuccessFlag;
	}
	
	public boolean isTrainSuccessFlag() {
		return trainSuccessFlag;
	}
	
	public void setTrainSuccessFlag(boolean trainSuccessFlag) {
		this.trainSuccessFlag = trainSuccessFlag;
	}

	@Override
	public Desc getDesc() {
		return desc;
	}
	
	@Override
	public Object getWmc() {
		return model;
	}
	
	@Override
	public boolean isReady() {
		return trainSuccessFlag;
	}

	@Override
	public void finishTraining() {
	}

	@Override
	public void resetup() {
	}
}