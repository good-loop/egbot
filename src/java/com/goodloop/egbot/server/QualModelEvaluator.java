package com.goodloop.egbot.server;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintWriter;
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
import com.winterwell.maths.stats.distributions.d1.MeanVar1D;
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
	 * load the evaluation set
	 * @param setFile 
	 * @return 
	 * @throws IOException
	 */
	public static Pair2<List<File>, Desc<List<File>>> loadFiles(List<File> evalFiles) throws IOException {	
		// TODO: check with DW that it's okay to get just the name of the 1st eval file or should i switch it back to it expecting just one eval file
		Desc testDesc = new Desc(evalFiles.get(0).getName(), List.class);
		testDesc.put("f", evalFiles.get(0));
		return new Pair2(evalFiles, testDesc);
	}
	
	/**
	 * load the evaluation set
	 * @param setFile 
	 * @return 
	 * @throws IOException
	 */
	public static Pair2<List<Map<String, Object>>, Desc<List<Map<String,Object>>>> loadEvalSet(File setFile) throws IOException {		
		Desc testDesc = new Desc(setFile.getName(), List.class);
		testDesc.put("f", setFile);
		
		ArrayList<Map<String, Object>> set = new ArrayList<Map<String, Object>>();
		
		Gson gson = new Gson();
		JsonReader jr = new JsonReader(FileUtils.getReader(setFile));
		jr.beginArray();
					
		while(jr.hasNext()) {
			Map<String, Object> qa = gson.fromJson(jr, Map.class);			
			set.add(qa);
		}
		return new Pair2(set, testDesc);
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
		List<File> evalFiles = experiment.getTestData();
		List<Map<String,String>> saved = new ArrayList();
		for (File evalFile : evalFiles) {
			List<Map<String, Object>> evalSet = loadEvalSet(evalFile).first;
			for (int i = 0; i < evalSet.size(); i++) {
				Map<String, Object> eg = evalSet.get(i);
				String question = (String) eg.get("question");
				String target = (String) eg.get("answer");
							
				String generated = model.sample(question, expectedAnswerLength);
				
				Map<String,String> temp = new ArrayMap<>(
					"question", question,
					"target", target,
					"generated", generated
				);			
				System.out.printf("Example of generated answer: %s\n\n", generated);
				saved.add(temp);
			}
		}
		saveToFile(saved);
	}

	private void saveToFile(List<Map<String, String>> saved) {
		Depot depot = Depot.getDefault();
		depot.put(experiment.getDesc(), saved);
		depot.flush();		
	}
	
}
