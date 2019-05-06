package com.goodloop.egbot.server;

import java.io.File;
import java.util.List;

import org.junit.Test;

import com.winterwell.depot.Depot;
import com.winterwell.depot.DepotConfig;
import com.winterwell.depot.Desc;
import com.winterwell.depot.IHasDesc;
import com.winterwell.maths.ITrainable.Supervised;
import com.winterwell.maths.stats.distributions.cond.Cntxt;
import com.winterwell.nlp.io.Tkn;

// structure of egbot tests:
// create model (either MarkovModel or LSTM)
// run model with certain config, format is runModel(modelType, trainFiles, evalFiles, trainFilter, evalFilter, numEpochs, preprocessing)

public class EvaluationTest {

	/**
	 * artifacts tagged egbot should save to the shared network drive
	 */
//	@Test
	public void testDepot() {		
		Depot depot = Depot.getDefault();
		DepotConfig config = depot.getConfig();
		System.out.println(config);
		Desc desc = new Desc("hw.txt", String.class);
		desc.setTag("egbot");
		File lp = depot.getLocalPath(desc);
		System.out.println(lp+" "+lp.getAbsolutePath());
		depot.put(desc, "Hello World");
		depot.flush();
		assert lp.toString().contains("egbot-learning-depot") : lp;
	}
	
	// TEST LSTM
	
//	@Test
	public void testPauliusSampleLTSM() throws Exception {
		LSTM lstm = new LSTM();						
		new EvaluatePredictions().runModel(lstm, "pauliusSample", "pauliusSample", 1, 1, 1, "None", "vocabPos");
	}
	
	
//	@Test
	public void test20LTSM() throws Exception {
		LSTM lstm = new LSTM();						
		new EvaluatePredictions().runModel(lstm, "MSE-20", "MSE-20", 1, 1, 1, "None", "vocabPos");
	}
	
//	@Test
	public void test100LTSM() throws Exception {
		LSTM lstm = new LSTM();						
		new EvaluatePredictions().runModel(lstm, "MSE-100", "MSE-20", 1, 1, 5, "None", "vocabPos");
	}
	
//	@Test
	public void testFullLTSM() throws Exception {
		LSTM lstm = new LSTM();						
		new EvaluatePredictions().runModel(lstm, "MSE-full", "MSE-full", 100, 100, 5, "None", "vocabPos");
	}
	
//	@Test
	public void testFullTrialLTSM() throws Exception { // MSE-full-trial AKA the old trained-model
		LSTM lstm = new LSTM();						
		new EvaluatePredictions().runModel(lstm, "MSE-full-trained", "MSE-20", 100, 100, 5, "None", "vocabPos");
	}

	
	// TEST MARKOV
	
//	@Test
	public void testPauliusSampleMarkov() throws Exception {
		MarkovModel mm = new MarkovModel();						
		new EvaluatePredictions().runModel(mm, "pauliusSample", "pauliusSample", 1, 1, 5, "None", "vocabPos");
	}
	
//	@Test
	public void test20Markov() throws Exception {
		MarkovModel mm = new MarkovModel(); 	
		new EvaluatePredictions().runModel(mm, "MSE-20", "MSE-20", 1, 1, 5, "None", "vocabPos");
	}
	
//	@Test
	public void test100Markov() throws Exception {
		MarkovModel mm = new MarkovModel(); 	
		new EvaluatePredictions().runModel(mm, "MSE-100", "MSE-20", 1, 1, 5, "None", "vocabPos");
	}
	
//	@Test
	public void testFullMarkov() throws Exception {
		MarkovModel mm = new MarkovModel(); 	
		new EvaluatePredictions().runModel(mm, "MSE-full", "MSE-full", 100, 100, 5, "None", "vocabPos");
	}

	/**
	 * if the same model and setup is used, depot should load the trained model the 2nd time (rather than train it from scratch)
	 * @throws Exception
	 */
//	@Test
	public void testSameMarkovRunTwice() throws Exception {
		MarkovModel mm = new MarkovModel(); 
		new EvaluatePredictions().runModel(mm, "MSE-20", "MSE-20", 100, 1, 5, "None", "vocabPos");
		assert mm.trainSuccessFlag;
		
		MarkovModel mm2 = new MarkovModel(); 
		new EvaluatePredictions().runModel(mm2, "MSE-20", "MSE-20", 100, 1, 5, "None", "vocabPos");
		assert mm2.loadSuccessFlag;
	}

}
