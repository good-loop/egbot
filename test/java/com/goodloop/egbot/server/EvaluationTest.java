package com.goodloop.egbot.server;

import java.io.File;

import org.junit.Test;

import com.winterwell.depot.Depot;
import com.winterwell.depot.DepotConfig;
import com.winterwell.depot.Desc;

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
	
//	@Test
	public void testLTSM() throws Exception {
		LSTM lstm = new LSTM();						
		new EvaluatePredictions().runModel(lstm, "MSE-20", "MSE-20");	
	}	

	@Test
	public void testMarkov() throws Exception {
		MarkovModel mm = new MarkovModel("MSE-full"); // TODO: redundant desc, will change this (see {@link com.goodloop.egbot.server.MarkovModel.MarkovModel()})
		new EvaluatePredictions().runModel(mm, "MSE-full", "MSE-full");	
	}

}
