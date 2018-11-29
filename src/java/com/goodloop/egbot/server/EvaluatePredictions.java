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
import com.winterwell.maths.ITrainable;
import com.winterwell.maths.stats.distributions.d1.MeanVar1D;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;
import com.winterwell.utils.time.RateCounter;
import com.winterwell.utils.time.TUnit;
import com.winterwell.utils.web.SimpleJson;
/**
 * To evaluate EgBot!
 * 
 * First run {@link ConstructEvaluationSet}
 * 
 * Then run this.
 * 
 * @author Irina
 *
 */
public class EvaluatePredictions {
	static ArrayList<Map<String, Object>> evalSet;
	/**
	 * Suggested number of words in answer
	 */
	static int expectedAnswerLength = 30;

	public static void main(String[] args) throws Exception {
		loadEvalSet();
		evaluateMarkov();
		System.out.println();
		evaluateLSTM();
	}

	/**
	 * load the evaluation set
	 * @throws IOException
	 */
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
	
	/**
	 * evaluate existing lstm model 
	 * by scoring the avg accuracy of log probabilities 
	 * and showing examples of generated output
	 * @throws Exception
	 */
	public static void DEPRECATEDevaluateLSTM() throws Exception {
		double avgScore = 0;
		
		System.out.println("Loading LSTM Model ...");
		int modelVersion = 240066;//epochs50, 625926;// requires passing the ckpt version for a trained model to use
		TrainLSTM lstm = new TrainLSTM(modelVersion); 	
		
		System.out.println("Loading Vocabulary ...");
//		int vocabVersion = lstm.loadAndInitVocab();
		int vocabVersion = 135802;// requires passing the ckpt version for a saved vocab to use
		lstm.loadVocab(vocabVersion);
		
		System.out.println("Scoring ...");
		for (int i = 0; i < evalSet.size(); i++) {
			String question = (String) evalSet.get(i).get("question");
			String target = (String) evalSet.get(i).get("question");
			avgScore += lstm.scoreAnswer(question, target);
			if(i%10==0) {          
				// quantitative evaluation
				System.out.printf("Avg score after %d evaluation examples: %f\n", i, avgScore/(i+1));
				
				// qualitative evaluation
				String generated = lstm.sampleSeries((String) evalSet.get(i).get("question"), expectedAnswerLength);
				System.out.printf("Example of generated answer: %s\n\n", generated);
			}
		}
	}
		
	IEgBotModel model;

	/**
	 * evaluate existing lstm model 
	 * by scoring the avg accuracy of log probabilities 
	 * and showing examples of generated output
	 * @throws Exception
	 */
	public void evaluateModel() throws Exception {
		MeanVar1D avgScore = new MeanVar1D();			
		// train?
		if ( ! model.isReady()) {
			evaluateModel2_train();
		}
		
		Log.d("Scoring ...");
		for (int i = 0; i < evalSet.size(); i++) {
			Map<String, Object> eg = evalSet.get(i);
			String question = (String) eg.get("question");
			String target = (String) eg.get("answer");
						
			double score = model.scoreAnswer(question, target);
			avgScore.train1(score);
			
			// quantitative evaluation
			if(i%10==0) {
				System.out.printf("Avg score after %d evaluation examples: %f\n", i, avgScore.getMean());
				
				// qualitative evaluation
				String generated = model.sampleSeries(question, expectedAnswerLength);
				System.out.printf("Example of generated answer: %s\n\n", generated);				
			}
		}
	}

	private void evaluateModel2_train() {
		model.resetup();
		//		train - load files, feed into model
		Map eg;
		model.train1(eg);
		model.finishTraining();
	} 	
	
}
