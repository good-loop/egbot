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
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.goodloop.egbot.EgbotConfig;
import com.winterwell.datascience.Experiment;
import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.maths.ITrainable;
import com.winterwell.maths.stats.distributions.cond.Cntxt;
import com.winterwell.maths.stats.distributions.cond.Sitn;
import com.winterwell.maths.stats.distributions.d1.MeanVar1D;
import com.winterwell.nlp.corpus.SimpleDocument;
import com.winterwell.nlp.io.SitnStream;
import com.winterwell.nlp.io.Tkn;
import com.winterwell.utils.containers.ArrayMap;
import com.winterwell.utils.containers.Pair2;
import com.winterwell.utils.io.FileUtils;
import com.winterwell.utils.log.Log;
import com.winterwell.utils.time.RateCounter;
import com.winterwell.utils.time.TUnit;
import com.winterwell.utils.web.SimpleJson;
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
	private void saveToFile(List<Map<String, String>> saved) {
		Depot depot = Depot.getDefault();

		EgBotResults results = experiment.getResults();
		results.generatedAnswers = (List) saved;

		depot.put(experiment.getDesc(), experiment);
		depot.flush();		
	}
	
}
