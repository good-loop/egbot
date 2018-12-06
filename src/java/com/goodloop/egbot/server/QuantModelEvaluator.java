package com.goodloop.egbot.server;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
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
		EgBotData testData = (EgBotData) experiment.getTestData();
		List<Map<String,String>> saved = new ArrayList();
		for(File file : testData.files) {
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
				// filter so as to evaluate only on test data
				if ( testData.filter.accept(c)) {
					c++;
					continue;
				}
				c++;
				// !TODO: do we still need ConstructEvalSet? 
				// or do we need to adopt a simple MSE train/test format 
				// (right now there is the default MSE one and the ConstructEvalSet)
				Map eg = gson.fromJson(jr, Map.class);			
				Boolean is_answered = (Boolean) eg.get("is_answered");
				if (!is_answered) continue;	
				String question = (String) eg.get("question");
				String target = (String) eg.get("answer");
				double score = model.scoreAnswer(question, target);
				avgScore.train1(score);
				
				if(c%100==0) {
					System.out.printf("Avg score after %d evaluation examples: %f\n", c, avgScore.getMean());			
				}		
			} 
			jr.close();			
		}	
		saveToFile(avgScore);
	}
	
	public Object DEPRECATEDevaluateDataPoint(Map eg, MeanVar1D avgScore) throws IOException {
		Boolean is_answered = (Boolean) eg.get("is_answered");
		if (!is_answered) return avgScore;	
		String question = (String) eg.get("question");
		String target = (String) eg.get("answer");
		double score = experiment.getModel().scoreAnswer(question, target);
		avgScore.train1(score);
		
		if(avgScore.getCount()%100==0) {
			System.out.printf("Avg score after %d evaluation examples: %f\n", avgScore.getCount(), avgScore.getMean());			
		}
		return avgScore;
	}
	
	/**
	 * save experiment evaluation results
	 * @param saved
	 */
	private void saveToFile(MeanVar1D avgScore) throws IOException {
		Depot depot = Depot.getDefault();
		depot.put(experiment.getDesc(), avgScore);
		depot.flush();
	}
	
}
