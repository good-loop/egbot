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
		
		EgBotExperiment e1 = new EgBotExperiment();
		e1.setModel(mm, trainedModelDesc);
		
		// Train
		if (trainedMarkov==null) {
			// do training
			List<File> trainData = e1.getTrainData(); 
			mm.resetup();
			mm.train(trainData); 
			mm.finishTraining();
		}
		
		// Test
		List<File> evalFilesMarkov = e1.getTestData();
		Pair2<List<File>, Desc<List<File>>> testDataMarkov = QualModelEvaluator.loadFiles(evalFilesMarkov);
		e1.setTestData(testDataMarkov.first, testDataMarkov.second);
		
		QualModelEvaluator qualMarkov = new QualModelEvaluator(e1);
		qualMarkov.evaluateModel();
		
		QuantModelEvaluator quantMarkov = new QuantModelEvaluator(e1);
		quantMarkov.evaluateModel();

		
		
		// LSTM
		TrainLSTM lstm = new TrainLSTM();				
		Desc<IEgBotModel> trainedLSTMDesc = lstm.getDesc();
		IEgBotModel trainedLSTM = Depot.getDefault().get(trainedLSTMDesc);
		
		EgBotExperiment e2 = new EgBotExperiment();
		e2.setModel(lstm, trainedLSTMDesc);
		
		// Train TODO: fix this (implement empty methods, question is are they needed?)
		if (trainedLSTM==null) {
			// do training
			List<File> trainData = e1.getTrainData();
			trainedLSTM.resetup();
			trainedLSTM.train(trainData); 
			trainedLSTM.finishTraining();
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
