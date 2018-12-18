package com.goodloop.egbot.server;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.goodloop.egbot.EgbotConfig;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.utils.containers.ArraySet;
import com.winterwell.utils.containers.Containers;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.time.RateCounter;
import com.winterwell.utils.time.TUnit;
import com.winterwell.utils.web.SimpleJson;
/**
 * DEPRECATED 
 * 
 * generates evaluation set from egbot data 
 * the eval set represents 10% 
 * 
 * saves outputs in /data/eval.json
 *
 */
public class ConstructEvaluationSet {
	static ArrayList<Map<String, Object>> evalSet;
	static Random probCounter;
	// TODO: eval file name should be set?
	static File evalSetFile;
	
	public static void main(String[] args) throws IOException {
		probCounter = new Random();
		probCounter.setSeed(42);
		evalSet = constructEvalSet(loadData());
		assert ! evalSet.isEmpty();
		saveData(evalSet);
	}

	/**
	 * 
	 * @param initialSet list of mappings with keys ("question", "answer") 
	 * @return 
	 * @throws IOException
	 */
	private static ArrayList<Map<String, Object>> constructEvalSet(ArrayList<Map<String, String>> initialSet) throws IOException {
		
		ArrayList<Map<String, Object>> set = new ArrayList<Map<String, Object>>();
		for (int i = 0; i < initialSet.size()-4; i++) {
			
			// copy question and answer pair e.g. [ { "question": "let ...", "answer": "well ..." } ]
			Map<String, Object> temp = new HashMap(initialSet.get(i));
			
			// also adding 4 wrong answers in the form of a string array e.g. [ { ..., "wrong": ["try ...", "maybe ...", "this ...", "if ..."] } ]
			String[] wrongs = new String[4];
			for (int j = 1; j <= 4; j++) {
				wrongs[j-1] = initialSet.get(i+j).get("answer");
			}
			temp.put("wrong", wrongs);
			set.add(temp);
		}
		
		System.out.printf("Evaluation set size: %d", set.size());
		return set;
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
			// Look for files of the name e.g. MathStackExchangeAPI_Part_3.json or .json.zip
			// Avoid zip/unzip dupes
			ArraySet<String> basefilenames = new ArraySet();
			List<File> allfiles = Arrays.asList(config.srcDataDir.listFiles());
			assert allfiles.size() != 0 : config.srcDataDir;
			files = Containers.filter(allfiles, f -> {
				if ( ! f.getName().startsWith("MathStackExchangeAPI_Part")) return false; 
				String bn = FileUtils.getBasename(FileUtils.getBasename(f));
				if (basefilenames.contains(bn)) return false;
				basefilenames.add(bn);
				return f.getName().endsWith(".json") || f.getName().endsWith(".json.zip");
			});
			assert ! files.isEmpty() : allfiles;
		}
		// always have the same ordering
		Collections.sort(files);
		assert ! files.isEmpty() : config.srcDataDir;
		RateCounter rate = new RateCounter(TUnit.MINUTE.dt);
		
		ArrayList<Map<String, String>> initialSet = new ArrayList<Map<String, String>>();
		for(File file : files) {
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
			while(jr.hasNext()) {//&& initialSet.size() < desiredEvalSetSize+4) {
				Map qa = gson.fromJson(jr, Map.class);			
				Boolean is_answered = (Boolean) qa.get("is_answered");
				if ( ! is_answered) {
					continue;
				}
				String question_body = (String) qa.get("body_markdown");
				double answer_count = (double) qa.get("answer_count");
				for (int j = 0; j < answer_count; j++) {					
					Boolean is_accepted = (Boolean) SimpleJson.get(qa, "answers", j, "is_accepted");
					if ( ! is_accepted) {
						continue;
					}
					String answer_body = SimpleJson.get(qa, "answers", 0, "body_markdown");
					Map<String, String> temp  = new HashMap<String, String>();
					temp.put("question", question_body);
					temp.put("answer", answer_body);
					// probabilistic counter, adds data point to testing batch 1 out of 10 times   
					if (probCounter.nextInt(10) == 0) {
						initialSet.add(temp);
					}
					c++;
					rate.plus(1);
					if (c % 1000 == 0) {
						System.out.println(c+" "+rate+"...");
					}					
				}			
			} 
		}
		assert ! initialSet.isEmpty();
		return initialSet;
	}	
	
	private static void saveData(ArrayList<Map<String, Object>> set) throws FileNotFoundException {
		Gson g = new Gson();
		String evalPath = System.getProperty("user.dir") + "/data/eval.json";	
		PrintWriter out = new PrintWriter(evalPath);
		out.print(g.toJson(set));
		out.close();
	}
}
