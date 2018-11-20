package com.goodloop.egbot.server;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.goodloop.egbot.EgbotConfig;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.time.RateCounter;
import com.winterwell.utils.time.TUnit;
import com.winterwell.utils.web.SimpleJson;

public class EvaluatePredictions {
	static ArrayList<Map<String, Object>> evalSet;
	int expectedAnswerLength = 30;

	public static void main(String[] args) throws IOException {
		loadEvalSet();
		evaluateMarkov();
		//evaluateLSTM();
	}

	public static void loadEvalSet() throws IOException {
		evalSet = new ArrayList<Map<String, Object>>();
		
		String evalPath = System.getProperty("user.dir") + "/data/eval.json";	
		Gson gson = new Gson();
		JsonReader jr = new JsonReader(FileUtils.getReader(new File(evalPath)));
		jr.beginArray();
					
		while(jr.hasNext()) {
			Map<String, Object> qa = gson.fromJson(jr, Map.class);			
			evalSet.add(qa);
		}
	}

	public static void evaluateMarkov() throws IOException {
		MarkovModel mm = new MarkovModel();
		mm.load();
		for (int i = 0; i < evalSet.size(); i++) {
			String question = (String) evalSet.get(i).get("question");
			String target = (String) evalSet.get(i).get("question");
			// TODO: evaluate answer
			// feed it the target & then score
			mm.scoreAnswer(question, target);
		}
	} 
	
	public void evaluateLSTM() throws Exception {
		TrainLSTM lstm = new TrainLSTM(); // requires passing the ckpt version for a specific  model		
		for (int i = 0; i < evalSet.size(); i++) {
			String prediction = lstm.sampleSeries((String) evalSet.get(i).get("question"), expectedAnswerLength);
			String target = (String) evalSet.get(i).get("question");
			// TODO: evaluate answer
		}
	} 	
}
