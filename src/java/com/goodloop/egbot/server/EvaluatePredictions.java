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
import com.winterwell.datascience.Experiment;
import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.gson.Gson;
import com.winterwell.gson.stream.JsonReader;
import com.winterwell.maths.ITrainable;
import com.winterwell.maths.stats.distributions.d1.MeanVar1D;
import com.winterwell.utils.IFilter;
import com.winterwell.utils.containers.Pair2;
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
 * @testedby {@link EvaluationTest}
 */
public class EvaluatePredictions {
	
	public void run() throws Exception {
			
		// Markov 
//		MarkovModel mm = new MarkovModel();		
//		mm.load();
//		runModel(mm);		
				
		// LSTM 
		LSTM lstm = new LSTM();				
		runModel(lstm);				
	}
	
	void runModel(IEgBotModel model) throws Exception {
		Desc<IEgBotModel> modelDesc = model.getDesc();

		// refresh cache?
		Depot.getDefault().remove(modelDesc);
		
		// set up experiment
		EgBotExperiment experiment = new EgBotExperiment();
		// set the model the experiment uses
		experiment.setModel(model, modelDesc);
		
		// set up filters (that decide train/test split)
		IFilter<Integer> trainFilter = n -> n % 100 != 1;
		IFilter<Integer> testFilter = n -> ! trainFilter.accept(n);
		// load the list of egbot files
		List<File> files = EgBotDataLoader.setup();

		// Train
		// set the train filter		
		EgBotData trainData = new EgBotData(files, trainFilter);
		// set the train data the experiment uses
		Desc<EgBotData> trainDataDesc = new Desc("MSE-data", EgBotData.class);
		trainDataDesc.put("use", "train");		
		experiment.setTrainData(trainData, trainDataDesc);
		// already trained?
		IEgBotModel trainedModel = Depot.getDefault().get(modelDesc);
		if (trainedModel==null) {
			// do training
			Log.d("Starting training ...");
			EgBotDataLoader.train(experiment);
			Depot.getDefault().put(modelDesc, experiment.getModel());
		} else {
			// replace the untrained with the trained
			Log.d("Using pre-trained model");
			experiment.setModel(trainedModel, modelDesc);
		}		
		
		// Test
		// set the test filter		
		EgBotData testData = new EgBotData(files, testFilter);
		// set the test data the experiment uses
		Desc<EgBotData> testDataDesc = new Desc("MSE-data", EgBotData.class);
		testDataDesc.put("use", "test");
		experiment.setTestData(testData, testDataDesc);
		
		// set up qualitative evaluator
		QualModelEvaluator qual = new QualModelEvaluator(experiment);
		// conduct evaluation
		if (true) {
			qual.evaluateModel();
		}
		
		// set up quantitative evaluator
		QuantModelEvaluator quant = new QuantModelEvaluator(experiment);
		// conduct evaluation
		quant.evaluateModel();
		
		// NB: the evaluator classes both save results		
		Log.i("Results at: "+Depot.getDefault().getLocalPath(experiment.getDesc()));
	}
	
}
