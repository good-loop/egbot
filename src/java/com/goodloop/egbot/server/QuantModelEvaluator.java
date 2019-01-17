package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.text.MessageFormat;
import java.util.ArrayList;
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
			ProbCounter counter = new ProbCounter();
			for (int j = 0; j < 4; j++) {
				int wrongIdx = i; 
				while (wrongIdx == i) {
					wrongIdx = counter.getC().nextInt(evalSet.size());
				} 				
				String wrongAns = (String) evalSet.get(wrongIdx).get("answer");
				answers.add(wrongAns);
			}						
			double score = model.scorePickBest(question, target, answers); // score will be 1 if correct guess, 0 if incorrect
			assert MathUtils.isFinite(score) : score+" from Q: "+question+" answer: "+target;
			avgScore.train1(score);
				
			// log update
			if(i%10==0) Log.i(MessageFormat.format("Avg score after {0} evaluation examples: {1}\n", i, avgScore.getMean()));	
		}
		// save final score
		saveToFile(avgScore);
		Log.i(MessageFormat.format("Percent of correct guesses: {0}", avgScore.getMean()));
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
	
}
