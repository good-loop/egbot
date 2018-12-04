package com.goodloop.egbot.server;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
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
	 * load egbot zenodo files and save them in trainingDataArray as list of qa paragraphs tokenised e.g. [ [ "let", "us", "suppose", ... ] ]
	 * @return trainingDataArray
	 * @throws IOException
	 */
	public List<List<String>> load() throws IOException {
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
	 * helper function to tokenise sentences
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
		String[] splitted = words.split("\\s+");
		return splitted;
	}
}
