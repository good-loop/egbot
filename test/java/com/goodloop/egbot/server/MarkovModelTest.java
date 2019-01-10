package com.goodloop.egbot.server;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.junit.Test;

import com.winterwell.depot.Depot;
import com.winterwell.depot.Desc;
import com.winterwell.utils.IFilter;
import com.winterwell.utils.log.Log;

/**
 * @author daniel
 *
 */
public class MarkovModelTest {

//	@Test
	public void testLoad() {
		Depot.getDefault().init();
		MarkovModel mm = new MarkovModel();
		mm.load();
	}

//	@Test
	public void testSave() {
		Depot.getDefault().init();
		MarkovModel mm = new MarkovModel();
		mm.save();
	}
	
	@Test
	public void testTrain() throws IOException {
		MarkovModel mm = new MarkovModel();
		Desc<IEgBotModel> modelDesc = mm.getDesc();

		// set up filters (that decide train/test split)
		IFilter<Integer> trainFilter = n -> n % 100 != 1;
		// load the list of egbot files
		List<File> files = EgBotDataLoader.setup();
		EgBotExperiment experiment = new EgBotExperiment();
		// set the model the experiment uses
		experiment.setModel(mm, modelDesc);

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
	public void testSample() throws IOException {
		String q = "what is a gaussian distribution";
		
		System.out.println("Loading model ...");
		MarkovModel mm = new MarkovModel();
		mm.load();  // TODO: clarify the diff between load and init for MM and LSTM
		
		System.out.println("Generating answer ...");
		String answer = mm.sample(q, 30);	
		System.out.println(answer);
	}
}
