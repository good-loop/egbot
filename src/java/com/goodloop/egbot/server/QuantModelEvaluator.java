package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.maths.stats.distributions.d1.MeanVar1D;
import com.winterwell.utils.MathUtils;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;
/**
 * Quantitative Model Evaluator
 * 
 * takes in evaluation inputs and calculates score of outputs from trained models 
 *
 * @author Irina
 *
 */
public class QuantModelEvaluator {

	final EgBotExperiment experiment;
	
	/**
	 * Can be reused
	 */
	public QuantModelEvaluator(EgBotExperiment experiment) {
		this.experiment = experiment;
	}
	
	/**
	 * evaluate model using MSE-20 data (**Needed: because the way the set up the quant evaluateModel requires at least 100 data points to be wasted, but the "MSE-20" data set only has 20 q&a pairs so it requires something different)
	 * by scoring the avg accuracy of log probabilities 
	 * and showing examples of generated output
	 * @param model must be trained
	 * @param testDataDesc 
	 * @param trainDataDesc 
	 * @throws Exception
	 */
	public void evaluateModelTinyData()	throws Exception {
		//DEBUGGING
		int oddCount = 0; // count of how many times we're getting -Inf score
		
		MeanVar1D avgScore = new MeanVar1D();
		IEgBotModel model = experiment.getModel();
		assert model != null;
		// train? no
		assert model.isReady(); 

		Log.d("Scoring ...");
		
		// construct evaluation data set
		EgBotData testData = (EgBotData) experiment.getTestData();
		ArrayList<Map<Object,Object>> evalSet = new ArrayList();
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
				if ( ! testData.filter.accept(c)) {
					c++;
					continue;
				}
				c++;
				Map<Object,Object> eg = gson.fromJson(jr, Map.class);		
				evalSet.add(eg);									
			} 
			jr.close();
		}	

		// select right and wrong answers and then score the evaluation set
		// TODO: shouldn't this choke if run on MSE-full eval dataset? loading all eval data in memory to then randomly pick wrong ans seems like a lot 
		// (but i ran it once and it was fine but i'm really surprised? but then again it was with MM not LSTM)
		// possible solution: should i go thru each one like i do or simply select 5 random numbers a bunch of times 
		// (but that might mean that not every answer gets to be a correct answer and some redundancy)
		for (int i = 0; i < evalSet.size(); i++) {
			Map<Object, Object> qa = evalSet.get(i);
			String question = (String) qa.get("question");
			String target = (String) qa.get("answer");
			
			// a list containing 1 correct answer and 4 wrong ones
			ArrayList<String> answers = new ArrayList<String>();
			// adding correct answer
			answers.add(target);		

			// probabilistic counter to determine a random selection of 4 wrong answers from the set
			MyRandom counter = new MyRandom();
			for (int j = 0; j < 4; j++) {
				int wrongIdx = i; 
				String wrongAns = (String) evalSet.get(wrongIdx).get("answer");
				int attempts = 0; // counter to avoid infinite loop
				while (wrongIdx == i || answers.indexOf(wrongAns)!=-1) { // they should all be unique (to avoid an answer array with the same answer four times + the right answer)
					attempts++;
					wrongIdx = counter.getC().nextInt(evalSet.size());			
					wrongAns = (String) evalSet.get(wrongIdx).get("answer");
					if(attempts>10*evalSet.size()) throw new Exception("Does the evaluation dataset have enough data points? I'm running out of 'wrong' answers to use in evaluation.");
				}
				answers.add(wrongAns);
			}	
			String bestAns = model.generateMostLikely(question,0);
			if(answers.indexOf(bestAns)==-1) {
				answers.add(bestAns);				
			} 
			double score = scorePickBest(model, question, target, answers); // score will be 1 if correct guess, 0 if incorrect
			if (!MathUtils.isFinite(score))
				oddCount += 1;
			assert MathUtils.isFinite(score) : score+" from Q: "+question+" answer: "+target;
			avgScore.train1(score);	
				
			// log update
			if(i%10==0) Log.i(MessageFormat.format("Avg score after {0} evaluation examples: {1}\n", i, avgScore.getMean()));	
		}
		// save final score
		saveToFile(avgScore);
		Log.i(MessageFormat.format("Percent of correct guesses: {0}", avgScore.getMean()*100));
		// DEBUGGING -Inf score
		Log.i(MessageFormat.format("Count of -Inf scores out of total: {0}/{1}\n", oddCount, avgScore.getCount()));	
}

	/**
	 * evaluate model 
	 * by scoring the avg accuracy of log probabilities 
	 * and showing examples of generated output
	 * @param eLabel 
	 * @param model must be trained
	 * @param testDataDesc 
	 * @param trainDataDesc 
	 * @throws Exception
	 */
	public void evaluateModel(String eLabel) throws Exception {
		final MyRandom counter = new MyRandom();
		
		// if eval set is too small, don't do random selection of wrong answers (**Needed: because the way the set up the quant evaluateModel requires at least 100 data points to be wasted, but for example the "MSE-20" data set only has 20 q&a pairs so it requires something different)
		if(eLabel.equals("MSE-20") || eLabel.equals("pauliusSample") || eLabel.equals("irinaSample") || eLabel.equals("statsBookJSON") || eLabel.equals("MSE-100")) {
			evaluateModelTinyData();
			return;
		}
		
		MeanVar1D avgScore = new MeanVar1D();
		IEgBotModel model = experiment.getModel();
		assert model != null;
		// train? no
		assert model.isReady(); 

		Log.d("Scoring ...");
		
		// construct evaluation data set
		EgBotData testData = (EgBotData) experiment.getTestData();
		ArrayList<Map<Object,Object>> evalSet = new ArrayList();
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
				Map<Object,Object> qa = gson.fromJson(jr, Map.class);
				c++;
				// filter so as to evaluate only on test data
				if ( ! testData.filter.accept(c) ) {
					continue;
				}						
				// NB: we do basically throw away 100 data points 
				// (since we don't evaluate them as target answers) 
				// for small eval data sets, it really doesn't make sense
				// TODO: one solution could be that we wrap back, using the last data points eval the beginning? 
				
				int evalSize = evalSet.size();
				int count = avgScore.getCount();
				// keep adding data points to the eval set until you have 100
				
				if (evalSet.size()<100) {
					// add latest data point to set of last 100
					evalSet.add(qa);
					continue;
				}
				
				// for each new data point, look at the last 100 points, randomly pick 4 wrong answers						
				String question = (String) qa.get("question");
				String target = (String) qa.get("answer");
				
				// a list containing 1 correct answer and 4 wrong ones
				ArrayList<String> answers = new ArrayList<String>();
				
				// adding correct answer
				answers.add(target);
				
				// probabilistic counter to determine a random selection of 4 wrong answers from the set				
				for (int j = 0; j < 4; j++) {
					int wrongIdx = j; 
					String wrongAns = (String) evalSet.get(wrongIdx).get("answer");
					int attempts = 0; // counter to avoid infinite loop
					while (wrongIdx == j || answers.indexOf(wrongAns)!=-1) { // they should all be unique (to avoid an answer array with the same answer four times + the right answer)
						attempts++;
						wrongIdx = counter.getC().nextInt(evalSet.size());			
						wrongAns = (String) evalSet.get(wrongIdx).get("answer");
						if(attempts>evalSet.size()) throw new Exception("Does the evaluation dataset have enough data points? I'm running out of 'wrong' answers to use in evaluation.");
					}
					answers.add(wrongAns);		
				}				
				Collections.shuffle(answers, counter.getC());				
				// evaluate the model 
				double score = scorePickBest(model, question, target, answers); // score will be 1 if correct guess, 0 if incorrect
				assert MathUtils.isFinite(score) : score+" from Q: "+question+" answer: "+target;
				avgScore.train1(score);
				Log.i(MessageFormat.format("Score: {0}", score));
					
				// log update
				if(c%100==1) Log.i(MessageFormat.format("Avg score after {0} evaluation examples: {1}\n", c, avgScore.getMean()));
				
				// update evalSet so that the oldest gets shoved out and the new ones comes in
				ArrayList<Map<Object,Object>> tempSet = new ArrayList(evalSet);
				tempSet.add(qa);
				tempSet.remove(0);
				evalSet = tempSet;
			} 						
			jr.close();
		}
		// save final score
		saveToFile(avgScore);
		Log.i(MessageFormat.format("Percent of correct guesses: {0}", avgScore.getMean()*100));
	}

	/**
	 * save experiment evaluation results
	 * @param saved
	 */
	private void saveToFile(MeanVar1D avgScore) throws IOException {
		Depot depot = Depot.getDefault();
		
		EgBotResults results = experiment.getResults();
		results.setAvgScore(avgScore);
		experiment.setResults(results);
		
		Desc expDesc = experiment.getDesc();		
		depot.put(experiment.getDesc(), experiment);
				
		Log.d("Results of quantitative experiment saved to: \n" + Depot.getDefault().getLocalPath(expDesc));
		depot.flush();
	}
	
	/**
	 * score model's ability to guess the correct answer from a set of answers
	 * @param q question 
	 * @param t target
	 * @param a answers
	 * @return 1 if correct, 0 if incorrect
	 * @throws IOException
	 */
	public int scorePickBest(IEgBotModel model, String q, String t, ArrayList<String> a) throws IOException {
		int bestAnsIdx = pickBest(model, q, a);
		int correct = a.indexOf(t);
		assert correct != -1 : t+" vs "+a;
		if (correct == bestAnsIdx) return 1; 
		else return 0;
	}
	
	/**
	 * returns index of answer that has best score (aka avg log prob)
	 */	
	public int pickBest(IEgBotModel model, String q, ArrayList<String> answers) throws IOException  {
		int currBestIndex = -1;
		double currBestScore = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < answers.size(); i++) {
			double temp = model.scoreAnswer(q, answers.get(i));
			if( temp > currBestScore) {
				currBestScore = temp;
				currBestIndex = i;
			}
		}
		return currBestIndex;
	}
	
}
