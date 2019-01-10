package com.goodloop.egbot.server;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.utils.IFilter;
import com.winterwell.utils.ReflectionUtils;
import com.winterwell.utils.log.Log;

public class LSTMTest {

	@Test
	public void testTrainDummy() throws Exception {
		LSTM lstm = new LSTM();
		Desc<IEgBotModel> modelDesc = lstm.getDesc();

		// set up filters (that decide train/test split)
		IFilter<Integer> trainFilter = n -> n % 100 != 1;
		// load the list of egbot files
		List<File> files = EgBotDataLoader.setupDummy();
		EgBotExperiment experiment = new EgBotExperiment();
		// set the model the experiment uses
		experiment.setModel(lstm, modelDesc);

		// Train
		// set the train filter		
		EgBotData trainData = new EgBotData(files, trainFilter);
		// set the train data the experiment uses
		Desc<EgBotData> trainDataDesc = new Desc("Dummy-data", EgBotData.class);
		trainDataDesc.put("use", "train");		
		experiment.setTrainData(trainData, trainDataDesc);
		
		// do training
		Log.d("Starting training ...");
		EgBotDataLoader.train(experiment);
		Depot.getDefault().put(modelDesc, experiment.getModel());
	}
	
//	@Test
	public void testSampleDummy() throws Exception {
		String q = "what is a gaussian distribution";
		
		System.out.println("Loading model ...");
		LSTM lstm = new LSTM();
		lstm.init(EgBotDataLoader.setupDummy());
		lstm.load(); 
		
		System.out.println("Generating answer ...");
		String answer = lstm.generateMostLikely(q, 30);	
		
		System.out.println(answer);
	}
}
