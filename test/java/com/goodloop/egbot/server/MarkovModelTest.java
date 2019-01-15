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
	
//	@Test
	public void testTrain() throws IOException {
		MarkovModel mm = new MarkovModel();
		Desc<IEgBotModel> modelDesc = mm.getDesc();

		// set up filters (that decide train/test split)
		IFilter<Integer> trainFilter = n -> n % 100 != 1;
		// load the list of egbot files   
		List<File> files = EgBotDataLoader.setupTiny(); // use dummy data
		EgBotExperiment experiment = new EgBotExperiment();
		// set the model the experiment uses
		experiment.setModel(mm, modelDesc);

		// Train
		// set the train filter		
		EgBotData trainData = new EgBotData(files, trainFilter);
		// set the train data the experiment uses
		Desc<EgBotData> trainDataDesc = new Desc("Egbot-data", EgBotData.class);
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
		
		Log.i("Loading model ...");
		MarkovModel mm = new MarkovModel();
		mm.load();  // TODO: clarify the diff between load and init for MM and LSTM
		
		Log.i("Generating results ...");
		Log.i("Question: " + q);
		String answer = mm.sample(q, 30);	
		Log.i("Answer: " + answer);
	}
	
//	@Test
	public void testLoading() {
		MarkovModel mm = new MarkovModel();
		mm.load();
		assert mm.isReady();
	}
	
	@Test
	public void trainAndSaveMoreThanOneModel() throws Exception {
		File saveLocation1;
		File saveLocation2; 
		
		{// first model we want to save
			MarkovModel mm = new MarkovModel();
			Desc<IEgBotModel> modelDesc = mm.getDesc();
	
			// set up filters (that decide train/test split)
			IFilter<Integer> trainFilter = n -> n % 100 != 1;
			IFilter<Integer> testFilter = n -> ! trainFilter.accept(n);
			// load the list of egbot files
			List<File> files = EgBotDataLoader.setupTiny();
			
			// set up experiment
			EgBotExperiment experiment = new EvaluatePredictions().trainExp(mm, modelDesc, trainFilter, files);
			
			saveLocation1 = Depot.getDefault().getLocalPath(modelDesc);
			Log.i("Results at: " + saveLocation1);
		}
		
		{// second model we want to save
			MarkovModel mm = new MarkovModel();
			Desc<IEgBotModel> modelDesc = mm.getDesc();
		
			// set up filters (that decide train/test split)
			IFilter<Integer> trainFilter = n -> n % 100 != 1;
			IFilter<Integer> testFilter = n -> ! trainFilter.accept(n);
			// load the list of egbot files
			List<File> files = EgBotDataLoader.setupTiny();
			
			// set up experiment
			EgBotExperiment experiment = new EvaluatePredictions().trainExp(mm, modelDesc, trainFilter, files);
			
			saveLocation2 = Depot.getDefault().getLocalPath(modelDesc);
			Log.i("Results at: " + saveLocation2);
		}
		assert !saveLocation1.equals(saveLocation2);
	}
}
