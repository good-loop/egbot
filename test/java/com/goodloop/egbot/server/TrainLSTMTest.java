package com.goodloop.egbot.server;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;

import com.winterwell.depot.Depot;

public class TrainLSTMTest {
	
	@Test
	public void testTrain() throws IOException {
		new TrainLSTM().train();
	}
	
	public void testWordsIntoFeatureVector() {	
		new TrainLSTM().wordsIntoFeatureVector("here is a question");
	}

}
