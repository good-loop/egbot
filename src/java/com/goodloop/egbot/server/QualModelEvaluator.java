package com.goodloop.egbot.server;

import java.io.File;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.utils.containers.ArrayMap;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;
/**
 * Qualitative Model Evaluator
 * 
 * takes in evaluation inputs and spits out generated outputs from trained models 
 * 
 * @author Irina
 *
 */
public class QualModelEvaluator {
	
	/**
	 * Suggested number of words in answer
	 */
	static int expectedAnswerLength = 30;

	static File qualSetFile = new File("data/build/qualEval.json");
	
	final EgBotExperiment experiment;
	
	/**
	 * Can be reused
	 */
	public QualModelEvaluator(EgBotExperiment experiment) {
		this.experiment = experiment;
	}
	
	/**
	 * evaluate model 
	 * by scoring the avg accuracy of log probabilities 
	 * and showing examples of generated output
	 * @throws Exception
	 */
	public void evaluateModel() throws Exception {
		IEgBotModel model = experiment.getModel();
		assert model != null;
		// train? no
		assert model.isReady(); 

		Log.d("Evaluating ...");
		EgBotData testData = (EgBotData) experiment.getTestData();
		List<Map<String,?>> saved = new ArrayList();
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
				Map eg = gson.fromJson(jr, Map.class);		
				String question = (String) eg.get("question");
				String target = (String) eg.get("answer");
				String generated = ((IEgBotModel) experiment.getModel()).generateMostLikely(question, expectedAnswerLength);
				
				Map<String,String> temp = new ArrayMap<>(
					"question", question,
					"target", target,
					"generated", generated
				);			
				System.out.printf("Example of generated answer: %s\n\n", generated);
				saved.add(temp);
				
			} 
			jr.close();			
		}	
		saveToFile(saved);
	}

	/**
	 * save experiment evaluation results
	 * @param saved
	 */
	private void saveToFile(List<Map<String, ?>> saved) {
		Depot depot = Depot.getDefault();

		EgBotResults results = experiment.getResults();
		results.setGeneratedAnswers(saved);
		experiment.setResults(results);

		Desc expDesc = experiment.getDesc();		
		depot.put(experiment.getDesc(), experiment);
		
		Log.d("Results saved to: " + Depot.getDefault().getLocalPath(expDesc));
		depot.flush();		
	}
	
}
