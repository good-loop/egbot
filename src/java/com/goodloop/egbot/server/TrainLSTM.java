package com.goodloop.egbot.server;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
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
public class TrainLSTM implements IEgBotModel {
	// model 
	List<Tensor<?>> model;
	public Desc desc;
	public boolean loadSuccessFlag;
	// TODO: when should trainSuccessFlag be true? 
	public boolean trainSuccessFlag;

	// input: training data and vocab
	HashMap<Integer, String> vocab;
	HalfLifeMap<String, Integer> hlVocab; 
	// TODO: remove redundancy of keeping both vocab var, this is prob left over from figuring out how to do top1000
	int idealVocabSize;
	int vocab_size;
	
	// training parameters
	int seq_length; // sequence length
	int num_epochs; // training epochs 
	int num_hidden; // number of hidden layers

	/**
	 * default constructor
	 * @throws IOException
	 */
	TrainLSTM() throws IOException{
		// TODO: is this the right way to identify model?
		model = new ArrayList<Tensor<?>>();
		desc = new Desc<>("MSE-slim", model.getClass());
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
		
		loadVocab(desc.getName(), files);
	}
	
	/**
	 * load egbot slim files and construct vocab (without saving training data because it's too memory consuming)
	 * @return magic version number -- needed to load
	 */
	public void loadAndInitVocab(List<File> files) throws IOException {
		// vocab has to be constructed and saved from all the text that will be used when training 
		// this is because vocab_size defines the shape of the feature vectors
		System.out.println("Loading files and initialising vocabulary");
		 	
		// construct vocab that auto-prunes and discards words that appear rarely
		// hlVocab is a map where the key to be the word and the value to be the word counts
		hlVocab = new HalfLifeMap<String, Integer>(idealVocabSize);
		
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
		String vocabPath = System.getProperty("user.dir") + "/data/models/final/v3/vocab_" + desc.getName() + ".txt";
		File vocabFile = new File(vocabPath);
		vocabFile.createNewFile(); 
		
		try (PrintWriter out = new PrintWriter(vocabFile)) {
			for (String word : hlVocab.keySet()) {
			    out.println(word);
			}
		}		
		System.out.printf("Saved vocab to file: %s\n", vocabPath);
		System.out.printf("Initialised vocabulary size: %s\n", hlVocab.size());
	}
	
	/**
	 * top SIZE words in the vocabulary (where size could be 1000 for the 1000th most common words)
	 */
	public ArrayList<String> vocabTop(int size) throws IOException {
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

			System.out.printf("Saved trained model to file: %s%s\n", "/data/models/final/v3/checkpoint", desc.getName());
		}
	}

	/**
	 * load egbot zenodo files and save them in trainingDataArray as list of qa paragraphs tokenised e.g. [ [ "let", "us", "suppose", ... ] ]
	 * @return trainingDataArray
	 * @throws IOException
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
					return name.startsWith("MathStackExchangeAPI_Part_1") && name.endsWith(".json");
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
				String question_body = (String) qa.get("question");
				String answer_body = (String) qa.get("answer");
				String[] temp = EgBotDataLoader.tokenise(question_body + " " + answer_body);
				trainingData.add(Arrays.asList(temp));
				c++;
				rate.plus(1);
				if (c % 1000 == 0) System.out.println(c+" "+rate+"...");
			} 
			jr.close();
		}
		return trainingData;
	}
	
	/**
	 * load vocabulary from file
	 * @param string see {@link #loadAndInitVocab()}
	 * @throws IOException
	 */
	public void loadVocab(String string, List<File> files) throws IOException{	
		
		// load the vocab to a map that allows for unique indexing of words
		// vocab is a map where the key is the unique index of the word, and the value is the word itself
	    vocab = new HashMap<Integer, String>();		
	    vocab.put(0,"UNKNOWN");
		vocab.put(1,"START");
		vocab.put(2,"END");
		vocab.put(3,"ERROR");
		int vocabIdx = 4;
		
		String vocabPath = System.getProperty("user.dir") + "/data/models/final/v3/vocab_" + String.valueOf(string) + ".txt";
        // checks to see if it finds the vocab file
	    final boolean checkpointExists = Files.exists(Paths.get(vocabPath));
	    if(!checkpointExists) {
	    	Log.d("Warning: Couldn't find vocab file");
	    	loadAndInitVocab(files); 
	    }
		File vocabFile = new File(vocabPath);
        try(BufferedReader br = new BufferedReader(new FileReader(vocabFile))) {
            for(String word; (word = br.readLine()) != null; ) {
    			vocab.put(vocabIdx, word);
    			vocabIdx += 1;
            }
        }
		vocab_size = vocab.size();
		System.out.printf("Loaded vocabulary size: %s\n", vocab_size);
	}

	/**
	 * initialise vocabulary using HalfLifeMap
	 * status: replaced by loadAndInitVocab (because it's inefficient to load the content from the files into memory and then init vocab separately)
	 * @throws IOException 
	 */
	@Deprecated
	public void initVocabHalfLifeMap(List<List<String>> trainingDataArray, int version) throws IOException{	
		// vocab has to be constructed and saved from all the text that will be used when training 
		// this is because vocab_size defines the shape of the feature vectors
		System.out.println("Initialising vocabulary");
		 		
		HalfLifeMap<String, Integer> hlVocab = new HalfLifeMap<String, Integer>(idealVocabSize);
		// construct vocab that auto-prunes and discards words that appear rarely
		// hlVocab is a map where the key to be the word and the value to be the word counts
		for (int i = 0; i < trainingDataArray.size(); i++) {
			if (i%100 == 0)	System.out.printf("In loop: %d out of %d\n Vocab size: %d\n\n", i, trainingDataArray.size(), hlVocab.size());
			List<String> qa = trainingDataArray.get(i);
			for (int j = 0; j < qa.size(); j++) {
				//System.out.printf(" In loop: %d out of %d\n", j, qa.size());
				String word = qa.get(j);
				//System.out.println(word);
				if (!word.equals("") && hlVocab.containsKey(word)) {
					int count = hlVocab.get(word) + 1;
					hlVocab.put(word, count);
				}
				else {
					hlVocab.put(word, 1);
				}
			}
		}		

		// save to file
		String vocabPath = System.getProperty("user.dir") + "/data/models/final/v3/vocab_v" + String.valueOf(version) + ".txt";
		File vocabFile = new File(vocabPath);
		vocabFile.createNewFile(); 
		
		try (PrintWriter out = new PrintWriter(vocabFile)) {
			for (String word : hlVocab.keySet()) {
			    out.println(word);
			}
		}		
		System.out.printf("Initialised vocabulary size: %s\n", hlVocab.size());
	}
	
	/**
	 * initialise vocabulary using HashMap
	 * status: replaced by loadAndInitVocab (because it's inefficient to load the content from the files into memory and then init vocab separately)
	 */
	@Deprecated
	private void initVocabHashMap(List<List<String>> trainingDataArray){	 
		// should I use a TreeSet instead of a HashMap, log(n) for basic operations and unique https://stackoverflow.com/questions/13259535/how-to-maintain-a-unique-list-in-java
		
		//HalfLifeMap<K, V>
		
		//Containers.getSortedKeys(map)
		
		System.out.println("Initialising vocabulary");
		
	    vocab = new HashMap<Integer, String>();		
	    vocab.put(0,"UNKNOWN");
		vocab.put(1,"START");
		vocab.put(2,"END");
		vocab.put(3,"ERROR");
		
		// construct vocab
		int vocabIdx = 4;
		for (int i = 0; i < trainingDataArray.size(); i++) {
			if (i%100 == 0)	System.out.printf("In loop: %d out of %d\n", i, trainingDataArray.size());
			List<String> qa = trainingDataArray.get(i);
			for (int j = 0; j < qa.size(); j++) {
				//System.out.printf(" In loop: %d out of %d\n", j, qa.size());
				if (!vocab.containsValue(qa.get(j))) {
					vocab.put(vocabIdx, qa.get(j));				
					vocabIdx += 1;
				}
			}
		}		
		vocab_size = vocab.size(); 
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
		final String checkpointDir = System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + desc.getName();
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
				//TODO: should we restore the model for every training step
				// an alternative would be to pass the sess var or define it as global and init it
			} else {
				System.out.println("Initialising model ...");
				sess.runner().addTarget("init").run();
			}
			
			// print out weight and bias initialisation
			//System.out.print("Starting from: \n");
			//printVariables(sess);
			
			ArrayList<Float> trainAccuracies = new ArrayList<Float>();
			ArrayList<Float> validAccuracies = new ArrayList<Float>();
			float trainAccuracy = 0;
			float validAccuracy = 0;
			
			// train a bunch of times
			// TODO: will it be more efficient if we sent batches instead of individual values?
			for (int epoch = 1; epoch <= num_epochs; epoch++) {
				
				// for each qa segment
				for (int qaIdx = 0; qaIdx < trainingDataArray.size(); qaIdx++) {
					List<String> temp = trainingDataArray.get(qaIdx);
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
						
						// print out training example
						if (epoch%100 == 1 && wordIdx == 30) {
							System.out.printf("epoch = %d qaIdx = %d wordIdx = %d \nInstance: %s\n Target: %s \n\n", 
									epoch, qaIdx, wordIdx, Arrays.deepToString(instanceArray), target);							
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
							
							trainAccuracies.add(runner.get(0).floatValue());							
			
							// close tensors to save memory
							closeTensors(runner);
						}
					}
				}
			}

			// save model checkpoint
			save(sess,checkpointPrefix);
	    }
	}

	/**
	 * saves trained model in the standard tensorflow checkpoint format
	 * @param sess
	 * @param checkpointPrefix
	 */
	private void save(Session sess, Tensor<?> checkpointPrefix) {
		model = sess.runner().feed("save/Const", checkpointPrefix).addTarget("save/control_dependency").run();
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
		final String checkpointDir = System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + desc.getName();
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
		final String checkpointDir = System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + desc.getName();
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
		final String checkpointDir = System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + desc.getName();
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
		String[] temp = EgBotDataLoader.tokenise(question_body + " " + answer_body);
		List<List<String>> trainingBatch = new ArrayList<List<String>>(); 
		trainingBatch.add(Arrays.asList(temp));
		// have to do a try block, because adding a throw declaration would mean i can't use the interface method train1() or i have to alter it
		try {
			trainEach(trainingBatch);
		} catch (IOException e) {
			e.printStackTrace();
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
		final String checkpointDir = System.getProperty("user.dir") + "/data/models/final/v3/checkpoint" + desc.getName();
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
		if(loadSuccessFlag && trainSuccessFlag) {
			return true;
		}
		return false;
	}
	
	/**
	 * reset cache
	 */
	public void reset() {
		Depot.getDefault().remove(desc);
	}

	@Override
	public void finishTraining() {
	}

	@Override
	public void resetup() {
	}	
}
