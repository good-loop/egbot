package com.goodloop.egbot.server;

public class TrainLSTMTest {
		
//	@Test
//	public void testInitVocab() throws IOException {
//		int ckptVersion = new Random().nextInt(1000000);
//		TrainLSTM lstm = new TrainLSTM(ckptVersion);
//		lstm.loadAndInitVocab();
//	}
	
//	@Test
//	public void testTrain() throws IOException {
//		int ckptVersion = new Random().nextInt(1000000);
//		TrainLSTM lstm = new TrainLSTM(ckptVersion);
//		int vocabVersion = 135802;
//		lstm.loadVocab(vocabVersion);
//		//lstm.checkTrainSize();
//		lstm.loadAndTrain();
//	}
	
//	@Test
//	public void testSampleSingleWord() throws Exception {
//		int ckptVersion = 432343;
//		TrainLSTM lstm = new TrainLSTM(ckptVersion);
//		int vocabVersion = 978513;
//		lstm.loadVocab(vocabVersion);
//		lstm.loadAndTrain();
//		lstm.sampleWord("what is probability");
//	}
//	
//	@Test
//	public void testSampleSeriesOfWords() throws Exception {
//		int ckptVersion = 432343;
//		TrainLSTM lstm = new TrainLSTM(ckptVersion);
//		int vocabVersion = 978513;
//		lstm.loadVocab(vocabVersion);
//		lstm.loadAndTrain();
//		lstm.sampleSeries("what is probability", 30);
//	}
	
//	@Test
//	public void testVocab10() throws Exception {
//		int ckptVersion = new Random().nextInt(1000000);
//		TrainLSTM lstm = new TrainLSTM(ckptVersion);
//		int vocabVersion = lstm.loadAndInitVocab();
//		lstm.loadVocab(vocabVersion);
//		lstm.vocabTop(10);
//	}	
	
	
//	@Test
//	public void testA() throws Exception {
//		// test constructing new vocab & training new model 
//		int ckptVersion = new Random().nextInt(1000000);
//		TrainLSTM lstm = new TrainLSTM(ckptVersion);
//		int vocabVersion = lstm.loadAndInitVocab();
//		lstm.loadVocab(vocabVersion);
//		lstm.loadAndTrain();
//		lstm.sampleSeries("what is probability", 30);
//	}
	
//	@Test
//	public void testB() throws Exception {
//		// test loading constructed vocab & training new model 
//		int ckptVersion = new Random().nextInt(1000000);
//		TrainLSTM lstm = new TrainLSTM(ckptVersion);
//		int vocabVersion = 857600; 
//		lstm.loadVocab(vocabVersion);
//		lstm.loadAndTrain();
//		lstm.sampleSeries("what is probability", 30);
//	}
	
//	@Test
//	public void testC() throws Exception {
//		// testing loading constructed vocab & loading trained model
//		int ckptVersion = 432343;
//		TrainLSTM lstm = new TrainLSTM(ckptVersion);
//		int vocabVersion = 978513;
//		lstm.loadVocab(vocabVersion);
//		lstm.loadAndTrain();
//		lstm.sampleSeries("what is probability", 30);
//	}

}
