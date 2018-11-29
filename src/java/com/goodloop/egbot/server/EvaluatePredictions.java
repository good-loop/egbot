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
	ArrayList<Map<String, Object>> evalSet;
	QuantModelEvaluator quant; 
	QualModelEvaluator qualt;
	
	public void main() throws Exception {
		// choose evaluation set 
		String evalSetPath = "/data/eval.json"; // default save location of ConstructEvaluationSet
		evalSet = loadEvalSet(evalSetPath);
		
		// Markov 
		MarkovModel mm = new MarkovModel();		
		mm.load();
		
		quant.evaluateModel(evalSet, mm);
		qualt.evaluateModel(evalSet, mm);
		
		// LSTM
		TrainLSTM lstm;		

		System.out.println("Loading LSTM Model ...");
		int modelVersion = 240066;//epochs50, 625926;// requires passing the ckpt version for a trained model to use
		lstm = new TrainLSTM(modelVersion); 	
		lstm.load();
		
		System.out.println("Loading Vocabulary ...");
		// int vocabVersion = lstm.loadAndInitVocab();
		int vocabVersion = 135802;// requires passing the ckpt version for a saved vocab to use
		lstm.loadVocab(vocabVersion);
		
		quant.evaluateModel(evalSet, lstm);
		qualt.evaluateModel(evalSet, lstm);
	}

	/**
	 * load the evaluation set
	 * @return 
	 * @throws IOException
	 */
	public ArrayList<Map<String, Object>> loadEvalSet(String location) throws IOException {
		ArrayList<Map<String, Object>> set = new ArrayList<Map<String, Object>>();
		
		String evalPath = System.getProperty("user.dir") + location;	
		Gson gson = new Gson();
		JsonReader jr = new JsonReader(FileUtils.getReader(new File(evalPath)));
		jr.beginArray();
					
		while(jr.hasNext()) {
			Map<String, Object> qa = gson.fromJson(jr, Map.class);			
			set.add(qa);
		}
		return set;
	}
}
