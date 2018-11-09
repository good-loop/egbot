package com.goodloop.egbot.server;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;

import com.winterwell.depot.Depot;

public class TrainLSTMTest {
	
//	@Test
//	public void testTrain() throws IOException {
//		TrainLSTM lstm = new TrainLSTM();
//		String result = lstm.train();
//		assertFalse(result.equals("<Error>"));
//	}
	
	@Test
	public void testSampleSeries() throws Exception {
		TrainLSTM lstm = new TrainLSTM();
		lstm.train();
		//String sample = lstm.sampleSeries(", the mice",4);
		//assertTrue(sample.equals("had a general council"));
	}
	
//	@Test
//	public void testWordsIntoFeatureVector() {	
//		new TrainLSTM().wordsIntoFeatureVector("here is a question");
//	}

}
