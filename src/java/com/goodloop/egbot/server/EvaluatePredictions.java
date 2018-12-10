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
 *
 */
public class EvaluatePredictions {
	
	public static void run() throws Exception {
			
		// Markov 
		MarkovModel mm = new MarkovModel();		
		Desc trainedModelDesc = mm.getDesc();
		//IEgBotModel trainedMarkov = Depot.getDefault().get(trainedModelDesc);
		mm.load();
		
		// set up experiment
		EgBotExperiment e1 = new EgBotExperiment();
		// set the model the experiment uses
		e1.setModel(mm, trainedModelDesc);
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
		e1.setTrainData(trainData, trainDataDesc);
		if (mm.getWmc()==null) {
			// do training
			EgBotDataLoader.train(e1);
		}
		
		// Test
		// set the test filter		
		EgBotData testData = new EgBotData(files, testFilter);
		// set the test data the experiment uses
		Desc<EgBotData> testDataDesc = new Desc("MSE-data", EgBotData.class);
		e1.setTestData(testData, testDataDesc);
		
		// set up qualitative evaluator
		QualModelEvaluator qualMarkov = new QualModelEvaluator(e1);
		// conduct evaluation
		qualMarkov.evaluateModel();

		// set up quantitative evaluator
		QuantModelEvaluator quantMarkov = new QuantModelEvaluator(e1);
		// conduct evaluation
		quantMarkov.evaluateModel();

		
		
		// LSTM 
		TrainLSTM lstm = new TrainLSTM();				
		Desc<IEgBotModel> trainedLSTMDesc = lstm.getDesc();
		lstm.reset(); // TODO: check that this is a valid way to refresh cache
		IEgBotModel trainedLSTM = Depot.getDefault().get(trainedLSTMDesc);
		
		// set up experiment
		EgBotExperiment e2 = new EgBotExperiment();
		// set the model the experiment uses
		e2.setModel(lstm, trainedLSTMDesc);

		// set up filters (that decide train/test split)
		IFilter<Integer> trainFilterLSTM = n -> n % 100 != 1;
		IFilter<Integer> testFilterLSTM = n -> ! trainFilterLSTM.accept(n);
		// load the list of egbot files
		List<File> filesLSTM = EgBotDataLoader.setup();

		// Train
		// set the train filter		
		EgBotData trainDataLSTM = new EgBotData(filesLSTM, trainFilterLSTM);
		// set the train data the experiment uses
		Desc<EgBotData> trainedDataDescLSTM = new Desc("MSE-data", EgBotData.class);
		e2.setTrainData(trainDataLSTM, trainedDataDescLSTM);
		if (trainedLSTM==null) {
			// do training
			System.out.println("Starting training ...");
			EgBotDataLoader.train(e2);
		}
		
		// Test
		// set the test filter		
		EgBotData testDataLSTM = new EgBotData(filesLSTM, testFilterLSTM);
		// set the test data the experiment uses
		Desc<EgBotData> testDataDescLSTM = new Desc("MSE-data", EgBotData.class);
		e2.setTestData(testDataLSTM, testDataDescLSTM);
		
		// set up qualitative evaluator
		QualModelEvaluator qualLSTM = new QualModelEvaluator(e2);
		// conduct evaluation
		qualLSTM.evaluateModel();
		
		// set up quantitative evaluator
		QuantModelEvaluator quantLSTM = new QuantModelEvaluator(e2);
		// conduct evaluation
		quantLSTM.evaluateModel();		
		
	}
}
