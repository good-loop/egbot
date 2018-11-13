package com.goodloop.egbot.server;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import com.goodloop.egbot.EgbotConfig;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.time.RateCounter;
import com.winterwell.utils.time.TUnit;
import com.winterwell.utils.web.SimpleJson;

public class ConstructEvaluationSet {
	static ArrayList<Map<String, String>> evalSet;
	static int desiredEvalSetSize = 100;
	
	public static void main(String[] args) throws IOException {
		constructEvalSet(loadData());
	}

	/**
	 * 
	 * @param initialSet list of mappings with keys ("question", "answer") 
	 * @throws IOException
	 */
	private static void constructEvalSet(ArrayList<Map<String, String>> initialSet) throws IOException {
		
		evalSet = new ArrayList<Map<String, String>>();
		for (int i = 0; i < initialSet.size()-4; i++) {
			
			if (evalSet.size() == desiredEvalSetSize) break;
			
			
			Map<String, String> temp = new HashMap(initialSet.get(i));
			//temp.put("question", temp.get("question"));
			//temp.put("right", temp.get("answer"));
			temp.put("wrong1", initialSet.get(i+1).get("answer"));
			temp.put("wrong2", initialSet.get(i+2).get("answer"));
			temp.put("wrong3", initialSet.get(i+3).get("answer"));
			temp.put("wrong4", initialSet.get(i+4).get("answer"));
			evalSet.add(temp);
			
//			for (Map.Entry<String, String> entry : temp.entrySet()) {
//			    System.out.println(entry.getKey()+" : "+entry.getValue());
//			}
		}
		System.out.printf("Evaluation set size: %d", evalSet.size());
	}	
	
	/**
	 * load egbot zenodo files and save them in inputData as list of q&a pairs e.g. [ [ "question ... ", "answer ..." ] ]
	 */
	private static ArrayList<Map<String, String>> loadData() throws IOException {
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
		
		ArrayList<Map<String, String>> initialSet = new ArrayList<Map<String, String>>();
		for(File file : files) {
			System.out.println("File: "+file+"...");
			Gson gson = new Gson();
			JsonReader jr = new JsonReader(FileUtils.getReader(file));
			jr.beginArray();
						
			int c=0;
			while(jr.hasNext() && initialSet.size() < desiredEvalSetSize*5) {
				Map qa = gson.fromJson(jr, Map.class);			
				Boolean is_answered = (Boolean) qa.get("is_answered");
				if (is_answered) {
					String question_body = (String) qa.get("body_markdown");
					double answer_count = (double) qa.get("answer_count");
					for (int j = 0; j < answer_count; j++) {					
						Boolean is_accepted = (Boolean) SimpleJson.get(qa, "answers", j, "is_accepted");
						if (is_accepted) {
							String answer_body = SimpleJson.get(qa, "answers", 0, "body_markdown");
							Map<String, String> temp  = new HashMap<String, String>();
							temp.put("question", question_body);
							temp.put("answer", answer_body);
							initialSet.add(temp);
							c++;
							rate.plus(1);
							if (c % 1000 == 0) System.out.println(c+" "+rate+"...");
						}
					}
				}			
			} 
		}
		return initialSet;
	}	
}
