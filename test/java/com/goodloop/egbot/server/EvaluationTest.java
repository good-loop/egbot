package com.goodloop.egbot.server;

import org.junit.Test;

public class EvaluationTest {

	@Test
	public void testLTSM() throws Exception {
		LSTM lstm = new LSTM();						
		new EvaluatePredictions().runModel(lstm);
	}	

	@Test
	public void testMarkov() throws Exception {
		MarkovModel lstm = new MarkovModel();						
		new EvaluatePredictions().runModel(lstm);
	}

}
