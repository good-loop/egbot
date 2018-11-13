package com.goodloop.egbot.server;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.Random;

import org.junit.Test;

import com.winterwell.depot.Depot;

public class TrainLSTMTest {
	
//	@Test
//	public void testInitVocab() throws IOException {
//		int ckptVersion = new Random().nextInt(1000000);
//		TrainLSTM lstm = new TrainLSTM(ckptVersion);
//		lstm.loadAndInitVocab();
//	}
	
	@Test
	public void testTrain() throws IOException {
		int ckptVersion = new Random().nextInt(1000000);
		TrainLSTM lstm = new TrainLSTM(ckptVersion);

		int vocabVersion = 957780;
		lstm.loadVocab(vocabVersion);
		lstm.loadAndTrain();
	}
	
//	@Test
//	public void testSampleSingleWord() throws Exception {
//		TrainLSTM lstm = new TrainLSTM(785346);
//		String sample = lstm.sampleSeries(", the mice",4);
//		assertTrue(sample.equals("had a general council"));
//	}
//	
//	@Test
//	public void testSampleSeriesOfWords() throws Exception {
//		TrainLSTM lstm = new TrainLSTM(785346);
//		String sample = lstm.sampleWord(", the mice");
//		assertTrue(sample.equals("had"));
//	}

}
