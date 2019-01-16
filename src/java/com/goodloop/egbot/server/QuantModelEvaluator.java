package com.goodloop.egbot.server;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.winterwell.depot.Depot;
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
		// TODO: should i do it this way or by simply selecting 5 random numbers a bunch of times (that might mean that not every answer gets to be a correct answer)
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
			
			//Log.i(MessageFormat.format(" Question: {0}\n Target: {1}\n Answers: {2}", question, target, answers.toString()));
			
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
		results.avgScore = avgScore;
		
		depot.put(experiment.getDesc(), experiment);
		
		Log.d("Results saved to: " + Depot.getDefault().getLocalPath(experiment.getDesc()));
		depot.flush();
	}
	
}
