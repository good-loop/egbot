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
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.maths.datastorage.HalfLifeMap;
import com.winterwell.nlp.io.ITokenStream;
import com.winterwell.nlp.io.StopWordFilter;
import com.winterwell.nlp.io.Tkn;
import com.winterwell.nlp.io.WordAndPunctuationTokeniser;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.time.RateCounter;
import com.winterwell.utils.time.TUnit;
import com.winterwell.utils.web.SimpleJson;
import com.goodloop.egbot.EgbotConfig;
public class EgBotDataLoader {
	List<File> files;
	RateCounter rate;
	
//	/**
//	 * load egbot zenodo files and save them in trainingDataArray as list of qa paragraphs tokenised e.g. [ [ "let", "us", "suppose", ... ] ]
//	 * @return trainingDataArray
//	 * @throws IOException
//	 */
//	private List<String> streamNext() {
//		List<String> trainingData;
//		for(File file : files) {
//			System.out.println("File: "+file+"...");
//			Gson gson = new Gson();
//			JsonReader jr = new JsonReader(FileUtils.getReader(file));
//			jr.beginArray();
//						
//			int c=0;
//			while(jr.hasNext()) {
//				Map qa = gson.fromJson(jr, Map.class);			
//				Boolean is_answered = (Boolean) qa.get("is_answered");
//				if (is_answered) {
//					String question_body = (String) qa.get("body_markdown");
//					double answer_count = (double) qa.get("answer_count");
//					for (int j = 0; j < answer_count; j++) {					
//						Boolean is_accepted = (Boolean) SimpleJson.get(qa, "answers", j, "is_accepted");
//						if (is_accepted) {
//							String answer_body = SimpleJson.get(qa, "answers", 0, "body_markdown");
//							trainingData = Arrays.asList(tokenise(question_body + " " + answer_body));
//							c++;
//							rate.plus(1);
//							if (c % 1000 == 0) System.out.println(c+" "+rate+"...");
//						}
//					}
//				}	
//			} 
//			jr.close();
//		}
//		return trainingData;		
//	}

//	public void startStream() {
//		EgbotConfig config = new EgbotConfig();
//		files = null;
//		if (false) {
//			// zenodo data slimmed down to filter only q&a body_markdown using python script data-collection/slimming.py
//			// Use this for extra speed if youve run the slimming script
//			// python script data-collection/slimming.py
//			files = Arrays.asList(new File(config.srcDataDir, "slim").listFiles());
//		} else {
//			files = Arrays.asList(config.srcDataDir.listFiles(new FilenameFilter() {				
//				@Override
//				public boolean accept(File dir, String name) {
//					return name.startsWith("MathStackExchangeAPI_Part_1") && name.endsWith(".json");
//				}
//			}));
//		}
//		// always have the same ordering
//		Collections.sort(files);
//		
//		rate = new RateCounter(TUnit.MINUTE.dt);
//	}
//
//	public void hasNext() {
//		// TODO Auto-generated method stub
//		
//	}
	
	/**
	 * finds egbot files in preparation for data loading
	 * @return list of egbot files
	 */
	public static List<File> setup() {
		List<File> files;
		EgbotConfig config = new EgbotConfig();
		assert config.srcDataDir.isDirectory() : config.srcDataDir;
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
	 * load egbot zenodo files and save them in trainingDataArray as list of qa paragraphs tokenised e.g. [ [ "let", "us", "suppose", ... ] ]
	 * @return trainingDataArray
	 * @throws IOException
	 */
	public static List<List<String>> load(List<File> files) throws IOException {
	
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
							trainingData.add(Arrays.asList(tokenise(question_body + " " + answer_body)));
							c++;
							rate.plus(1);
							if (c % 1000 == 0) System.out.println(c+" "+rate+"...");
						}
					}
				}	
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
		// TODO: fix this to remove stop words?
		String[] splitted = words.split("\\s+");
		return splitted;
	}
	
	/**
	 * a common train-over-MSE-data class
	 * @throws IOException
	 */
	public static void train(EgBotData trainData, IEgBotModel model) throws IOException {
		
		assert model.getWmc() != null;
		// load if we can
		model.load();
		// already done?
		if (model.getLoadSuccessFlag()) {
			return;
		}
		// no -- train!
		assert model.getWmc() != null;
					
		RateCounter rate = new RateCounter(TUnit.MINUTE.dt);
		
		for(File file : trainData.files) {
			System.out.println("File: "+file+"...");
	
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
				Boolean is_answered = (Boolean) qa.get("is_answered");
				if (!is_answered) continue;
				model.train1(qa);
				rate.plus(1);
				if (c % 1000 == 0) System.out.println(c+" "+rate+"...");
			} 
			jr.close();
			
//			if (false) break;
		}
		// save trained model
		model.save();
	}
	
}
