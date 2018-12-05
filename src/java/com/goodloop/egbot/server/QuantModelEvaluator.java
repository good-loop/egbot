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
import com.winterwell.datascience.Experiment;
import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
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
 * Quantitative Model Evaluator
 * 
 * takes in evaluation inputs and calculates score of outputs from trained models 
 *
 * @author Irina
 *
 */
public class QuantModelEvaluator {

	static File quantSetFile = new File("data/build/quantEval.json");
	
	final EgBotExperiment experiment;
	
	/**
	 * Can be reused
	 */
	public QuantModelEvaluator(EgBotExperiment experiment) {
		this.experiment = experiment;
	}

	/**
	 * evaluate model 
	 * by scoring the avg accuracy of log probabilities 
	 * and showing examples of generated output
	 * @param model must be trained
	 * @param testDataDesc 
	 * @param trainDataDesc 
	 * @throws Exception
	 */
	public void evaluateModel()	throws Exception {
		MeanVar1D avgScore = new MeanVar1D();			
		IEgBotModel model = experiment.getModel();
		assert model != null;
		// train? no
		assert model.isReady(); 

		Log.d("Scoring ...");
		// !TODO: fix this in the same way as QualModelEvaluator
		List<File> evalFiles = experiment.getTestData();
		for (File evalFile : evalFiles) {
			List<Map<String, Object>> evalSet = QualModelEvaluator.loadEvalSet(evalFile).first;
			for (int i = 0; i < evalSet.size(); i++) {
				Map<String, Object> eg = evalSet.get(i);
				String question = (String) eg.get("question");
				String target = (String) eg.get("answer");
							
				double score = model.scoreAnswer(question, target);
				avgScore.train1(score);
				
				if(i%100==0) {
					System.out.printf("Avg score after %d evaluation examples: %f\n", i, avgScore.getMean());			
				}
			}
		}
		saveToFile(avgScore);
	}
	
	private void saveToFile(MeanVar1D avgScore) throws IOException {
		Depot depot = Depot.getDefault();
		depot.put(experiment.getDesc(), avgScore);
		depot.flush();
	}
	
}
