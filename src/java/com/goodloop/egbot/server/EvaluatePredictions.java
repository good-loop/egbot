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
	
	public void run() throws Exception {
			
		// Markov TODO: check that this works
		MarkovModel mm = new MarkovModel();		
		Desc<IEgBotModel> trainedModelDesc = mm.getDesc();
		IEgBotModel trainedMarkov = Depot.getDefault().get(trainedModelDesc);
		
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
		if (trainedMarkov==null) {
			// do training
			EgBotDataLoader.train(trainData, mm);
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

		
		
		// LSTM !TODO: fix this in the same way as MM
		TrainLSTM lstm = new TrainLSTM();				
		Desc<IEgBotModel> trainedLSTMDesc = lstm.getDesc();
		IEgBotModel trainedLSTM = Depot.getDefault().get(trainedLSTMDesc);
		
		EgBotExperiment e2 = new EgBotExperiment();
		e2.setModel(lstm, trainedLSTMDesc);
		
		// Train
		if (trainedLSTM==null) {
			// do training
			EgBotDataLoader.train(trainData, lstm);
		}
		
		// Test
		List<File> evalFilesLSTM = e2.getTestData();
		Pair2<List<File>, Desc<List<File>>> testDataLSTM = QualModelEvaluator.loadFiles(evalFilesLSTM);
		e1.setTestData(testDataLSTM.first, testDataLSTM.second);
		
		QualModelEvaluator qualLSTM = new QualModelEvaluator(e2);
		qualLSTM.evaluateModel();
		
		QuantModelEvaluator quantLSTM = new QuantModelEvaluator(e2);
		quantLSTM.evaluateModel();		
		
	}
}
