package com.goodloop.egbot.server;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintWriter;
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
 * 
 * takes in evaluation inputs and calculates score of outputs from trained models 
 *
 * @author Irina
 *
 */
public class QuantModelEvaluator {
	/**
	 * Suggested number of words in answer
	 */
	int expectedAnswerLength = 30;
	/**
	 * Model to evaluate
	 */
	IEgBotModel model;

	/**
	 * evaluate model 
	 * by scoring the avg accuracy of log probabilities 
	 * and showing examples of generated output
	 * @throws Exception
	 */
	public void evaluateModel(ArrayList<Map<String, Object>> evalSet, IEgBotModel model) throws Exception {
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
			
			if(i%100==0) {
				System.out.printf("Avg score after %d evaluation examples: %f\n", i, avgScore);			
			}			
			
			if (i == evalSet.size()-1) saveToFile(avgScore.getMean()); 
		}
	}

	private void evaluateModel2_train() {
		Log.d("Training ...");

//		model.resetup();
//		//		train - load files, feed into model
//		Map eg;
//		model.train1(eg);
//		model.finishTraining();
	} 	
	
	private void saveToFile(double score) throws IOException {
		String resultsFileName = "smallTestQuantitativeEval.txt";
		String resultsPath = System.getProperty("user.dir") + "/data/models/final/v3/" + resultsFileName;
		File resultsFile = new File(resultsPath);
		resultsFile.createNewFile(); 
		
		try (PrintWriter out = new PrintWriter(resultsFile)) {
			out.printf("Avg score after %d evaluation examples: %f", score);
		}	
	}
	
}
