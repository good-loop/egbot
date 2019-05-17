package com.goodloop.egbot.server;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.goodloop.egbot.EgbotConfig;
import com.winterwell.datascience.Experiment;
import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.nlp.io.ITokenStream;
import com.winterwell.nlp.io.StopWordFilter;
import com.winterwell.nlp.io.Tkn;
import com.winterwell.nlp.io.WordAndPunctuationTokeniser;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.time.RateCounter;
import com.winterwell.utils.time.TUnit;
import com.winterwell.utils.log.Log;

public class EgBotDataLoader {
	List<File> files;
	RateCounter rate;
	String evalName;
	
	/**
	 * finds egbot files in preparation for data loading
	 * @return list of egbot files
	 */
	@Deprecated
	public static List<File> setupOld() {
		List<File> files;
		EgbotConfig config = new EgbotConfig();
		assert config.srcDataDir.isDirectory() : config.srcDataDir+" -- maybe run python slimAndTrim.py";
		if (false) {
			// zenodo data slimmed down to filter only q&a body_markdown using python script data-collection/slimming.py
			// Use this for extra speed if youve run the slimming script
			// python script data-collection/slimming.py
			files = Arrays.asList(new File(config.srcDataDir, "slim").listFiles());
		} else {
			File[] fs = config.srcDataDir.listFiles(new FilenameFilter() {				
				@Override
				public boolean accept(File dir, String name) {
					return name.startsWith("MathStackExchangeAPI_Part")  
							&& (name.endsWith(".json") || name.endsWith(".json.zip"));
				}
			});
			assert fs != null && fs.length > 0 : config.srcDataDir;
			files = Arrays.asList(fs);
		}
		// always have the same ordering
		Collections.sort(files);
		return files;
	}
	
	/**
	 * finds tiny files in preparation for data loading
	 * @return list of tiny files
	 */
	@Deprecated
	public static List<File> setupTiny() {
		List<File> fs = Arrays.asList(new File(System.getProperty("user.dir") + "/data/test_input/tiny.json"));
		return fs; 
	}	
	
	
	/**
	 * finds files in preparation for data loading
	 * @return list of files
	 */
	public static List<File> setup(String dataLabel) {
		List<File> files = null;
		
		switch (dataLabel) {
    		case "MSE-part":
        		files = loadMSE("MathStackExchangeAPI_Part_1"); // load MSE data that starts with this string (aka only first part of egbot data)
        		break;
        	case "MSE-full":
        		files = loadMSE("MathStackExchangeAPI_Part_1"); // load MSE data that starts with this string (aka all egbot data)
            	break;
        	case "MSE-20": 
        		files = Arrays.asList(new File(System.getProperty("user.dir") + "/data/test_input/tiny.json")); // 20 pre-selected MSE q&a pairs
        		break;
        	case "paul-20": 
        		files = Arrays.asList(new File(System.getProperty("user.dir") + "/data/test_input/paulius20.json")); // paulius' 20 questions (TODO: needs answers, currently has dummy ones)
        		break;        
		}
		// always have the same ordering
		Collections.sort(files);
		return files; 
	}
		
	/**
	 * load MSE data that starts with the string passed as param 
	 * @param fileNameStart
	 * @return
	 */
	private static List<File> loadMSE(String fileNameStart) {
		List<File> files = null;
		EgbotConfig config = new EgbotConfig();
		assert config.srcDataDir.isDirectory() : config.srcDataDir+" -- maybe run python slimAndTrim.py";
		File[] fs = config.srcDataDir.listFiles(new FilenameFilter() {				
			@Override
			public boolean accept(File dir, String name) {
				return name.startsWith(fileNameStart)  
						&& (name.endsWith(".json") || name.endsWith(".json.zip"));
			}
		});
		assert fs != null && fs.length > 0 : config.srcDataDir;
		files = Arrays.asList(fs);
		return files;
	}

	/**
	 * load egbot zenodo files and save them in trainingDataArray as list of qa paragraphs tokenised e.g. [ [ "let", "us", "suppose", ... ] ]
	 * @return trainingDataArray
	 * @throws IOException
	 */
	public static List<List<String>> load(List<File> files) throws IOException {
		RateCounter rate = new RateCounter(TUnit.MINUTE.dt);
		
		List<List<String>> trainingData = new ArrayList<List<String>>(); 
		for(File file : files) {
			Log.i("File: "+file+"...");
			Gson gson = new Gson();
			JsonReader jr = new JsonReader(FileUtils.getReader(file));
			jr.beginArray();
						
			int c=0;
			while(jr.hasNext()) {
				Map qa = gson.fromJson(jr, Map.class);			
				String question_body = (String) qa.get("question");
				String answer_body = (String) qa.get("answer");
				trainingData.add(Arrays.asList(tokenise(question_body + " " + answer_body)));
				c++;
				rate.plus(1);
				if (c % 1000 == 0) Log.i(c+" "+rate+"...");	
			} 
			jr.close();
		}
		return trainingData;
	}
	
	/**
	 * helper function to tokenise sentences (deprecated because StopWordFilter removes too many words)
	 * @param words
	 * @return
	 */
	@Deprecated
	public String[] DEPRECATEDtokenise(String words) {
		WordAndPunctuationTokeniser t = new WordAndPunctuationTokeniser();
		t.setSwallowPunctuation(true);
		t.setLowerCase(true);
		
		ITokenStream _tokeniser = t;
		_tokeniser = _tokeniser.factory(words);
		StopWordFilter s = new StopWordFilter(_tokeniser);
		List<Tkn> tknised = s.toList();
		
		String[] array = new String[tknised.size()];
		int index = 0;
		for (Tkn tk : tknised) {
		  array[index] = tk.getText();
		  index++;
		}
		return array;
	}
	
	/**
	 * helper function to tokenise sentences
	 * @param words
	 * @return
	 */
	public static String[] tokenise(String words) {
		String[] splitted = words.split("\\s+");
		return splitted;
	}
	
	/**
	 * a common train-over-MSE-data class. This will load the model from Depot if it can -- replacing the object in Experiment
	 * Does NOT save the model
	 * @throws IOException
	 */
	public static void train(Experiment e) throws IOException {
		EgBotData trainData = (EgBotData) e.getTrainData();
		IEgBotModel model = (IEgBotModel) e.getModel();
 
		Desc<IEgBotModel> modelDesc = model.getDesc();
		// Do we have a pre-trained version?
		IEgBotModel pretrained = Depot.getDefault().get(modelDesc);
		if (pretrained!=null) {
			// replace the untrained with the trained
			Log.d("Using pre-trained model");
			model = pretrained;
			e.setModel(pretrained, modelDesc);
			// TODO: remove temporary loadSuccessFlag flag and associated methods, this is just while I test model loading (can't hold loadSuccessFlag within model because of how we load the model)
			model.setLoadSuccessFlag(true);
		}

		// call load
		//model.load();

		// already done?
		if (model.isReady()) {
			return;
		}
		
		// no -- train!
		Log.i("Starting training ...");
		
		// init model (vocab etc)
		model.init(trainData.files, e.getNumEpoch(), e.getPreprocessing(), e.getWordEmbedMethod());
							
		RateCounter rate = new RateCounter(TUnit.MINUTE.dt);
		
		for(File file : trainData.files) {
			Log.i("File: "+file+"...");
	
			Gson gson = new Gson();
			// zip or plain json?
			Reader r;
			if (file.getName().endsWith(".zip")) {
				r = FileUtils.getZIPReader(file);
			} else {
				r = FileUtils.getReader(file);
			}
			JsonReader jr = new JsonReader(r);
			jr.beginArray();
						
			int c=-1;
			while(jr.hasNext()) {
				c++;
				Map qa = gson.fromJson(jr, Map.class);
				if ( ! trainData.filter.accept(c)) {
					continue;
				}			
				model.train1(qa);
				rate.plus(1);
				if (c % 1000 == 0) Log.i(c+" "+rate+"...");
			} 
			jr.close();
			
			if (false) break;
		}
		
		Log.i("Yay, finished training :) \n");		
		model.finishTraining();
		model.setTrainSuccessFlag(true); // ?? ideally move into finishTraining() but who cares
		Depot.getDefault().put(modelDesc, model);
		Depot.getDefault().flush();
	}

	public String getEvalName() {
		return evalName;
	}
	
}
