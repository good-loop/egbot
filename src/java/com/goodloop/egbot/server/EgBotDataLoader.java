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
	public static List<File> setup() {
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
        	case "MSE-full":
        		EgbotConfig config = new EgbotConfig();
        		assert config.srcDataDir.isDirectory() : config.srcDataDir+" -- maybe run python slimAndTrim.py";
        		File[] fs = config.srcDataDir.listFiles(new FilenameFilter() {				
    				@Override
    				public boolean accept(File dir, String name) {
    					return name.startsWith("MathStackExchangeAPI_Part")  
    							&& (name.endsWith(".json") || name.endsWith(".json.zip"));
    				}
    			});
    			assert fs != null && fs.length > 0 : config.srcDataDir;
    			files = Arrays.asList(fs);
            	break;
        	case "MSE-20": 
        		files = Arrays.asList(new File(System.getProperty("user.dir") + "/data/test_input/tiny.json"));
        		break;
        	case "paul-20": 
        		files = Arrays.asList(new File(System.getProperty("user.dir") + "/data/test_input/paulius20.json"));
        		break;        
		}
		// always have the same ordering
		Collections.sort(files);
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
	 * a common train-over-MSE-data class
	 * Does NOT save the model
	 * @throws IOException
	 */
	public static void train(Experiment e) throws IOException {
		EgBotData trainData = (EgBotData) e.getTrainData();
		IEgBotModel model = (IEgBotModel) e.getModel();

		assert model.getWmc() != null;
		// load if we can
		model.load();

		// already done?
		if (model.isReady()) {
			return;
		}
		
		// no -- train!
		assert model.getWmc() != null;
		
		Log.i("Starting training ...");
		
		// init model (vocab etc)
		model.init(trainData.files);
							
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
						
			int c=0;
			while(jr.hasNext()) {
				if ( ! trainData.filter.accept(c)) {
					c++;
					continue;
				}
				c++;
				
				Map qa = gson.fromJson(jr, Map.class);
				model.train1(qa);
				rate.plus(1);
				if (c % 1000 == 0) Log.i(c+" "+rate+"...");
			} 
			jr.close();
			
			if (false) break;
		}
		
		Log.i("Yay, finished training :) \n");
		model.setTrainSuccessFlag(true);
		Depot.getDefault().flush();
	}

	public String getEvalName() {
		return evalName;
	}
	
}
