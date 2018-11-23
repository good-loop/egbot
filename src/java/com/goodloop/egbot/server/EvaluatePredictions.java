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
		double avgScore = 0;
		MarkovModel mm = new MarkovModel();
		System.out.println("Training Markov Model ...");
		mm.train();
		//mm.load();
		System.out.println("Scoring Model ...");
		for (int i = 0; i < evalSet.size(); i++) {
			String question = (String) evalSet.get(i).get("question");
			String target = (String) evalSet.get(i).get("question");
			avgScore += mm.scoreAnswer(question, target);
			if(i%10000==0) {                                                                                                                                                                                                                                                           
				System.out.printf("Avg score after %d examples: %f\n", i, avgScore/(i+1));
			}
		}
	} 
	
	public void evaluateLSTM() throws Exception {
		double avgScore = 0;
		TrainLSTM lstm = new TrainLSTM(); // requires passing the ckpt version for a specific  model		
		for (int i = 0; i < evalSet.size(); i++) {
			String question = (String) evalSet.get(i).get("question");
			//String prediction = lstm.sampleSeries((String) evalSet.get(i).get("question"), expectedAnswerLength);
			String target = (String) evalSet.get(i).get("question");
			lstm.scoreAnswer(question, target);
			avgScore += lstm.scoreAnswer(question, target);
			if(i%10000==0) {                                                                                                                                                                                                                                                           
				System.out.printf("Avg score after %d examples: %f\n", i, avgScore/(i+1));
			}
		}
	} 	
}
